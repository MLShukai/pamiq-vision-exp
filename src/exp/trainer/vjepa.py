import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing import Value

import torch
from torch import Tensor
from torch.utils.data.dataloader import default_collate


@dataclass(frozen=True)
class MaskConfig:
    """Configuration for a mask type (short-range or long-range).

    Attributes:
        num_blocks: Number of blocks to sample for this mask type.
        spatial_scale: Target spatial coverage ratio for each individual block.
            For example, 0.15 means each block covers ~15% of the frame area.
    """

    num_blocks: int
    spatial_scale: float

    def __post_init__(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If parameters are invalid.
        """
        if self.num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")
        if not 0 < self.spatial_scale <= 1:
            raise ValueError("spatial_scale must be in (0, 1]")


class VideoMultiBlockMaskCollator:
    """V-JEPA collator for creating 3D multi-block mask tensors.

    Creates boolean masks where spatial blocks are extended across the entire
    temporal dimension (tube masking). Supports V-JEPA's multi-mask strategy
    with short-range and long-range masks.

    The masks are boolean tensors where:
    - True values indicate tubelets to be masked (ignored by context encoder)
    - False values indicate tubelets to be processed

    Following V-JEPA paper defaults (Section 3.2, Table 8):
    - Short-range: 8 blocks, each covering ~15% of frame
    - Long-range: 2 blocks, each covering ~70% of frame
    - Aspect ratio: (0.75, 1.5)
    - Temporal coverage: 100% (tubes extend full temporal dimension)
    - Combined masking ratio: ~90% (due to overlapping blocks)
    """

    def __init__(
        self,
        num_tubelets: tuple[int, int, int],
        mask_configs: list[MaskConfig] | None = None,
        aspect_ratio: tuple[float, float] = (0.75, 1.5),
        min_keep: int = 4,
    ) -> None:
        """Initialize the VideoMultiBlockMaskCollator.

        Args:
            num_tubelets: Number of tubelets as (n_temporal, n_height, n_width).
            mask_configs: List of mask configurations. Defaults to V-JEPA's
                short-range and long-range configs.
            aspect_ratio: Range of aspect ratios for spatial blocks.
            min_keep: Minimum number of spatial positions to keep unmasked.

        Raises:
            ValueError: If parameters are invalid.
        """
        if aspect_ratio[0] > aspect_ratio[1]:
            raise ValueError("aspect_ratio[0] must be <= aspect_ratio[1]")
        if aspect_ratio[0] <= 0:
            raise ValueError("aspect_ratio[0] must be positive")
        if min_keep < 1:
            raise ValueError("min_keep must be at least 1")

        self._n_t, self._n_h, self._n_w = num_tubelets
        self._aspect_ratio = aspect_ratio
        self._min_keep = min_keep

        if self._n_h * self._n_w <= min_keep:
            raise ValueError(
                f"Total spatial patches ({self._n_h * self._n_w}) "
                f"must exceed min_keep ({min_keep})"
            )

        self._mask_configs = mask_configs or [
            MaskConfig(num_blocks=8, spatial_scale=0.15),  # short-range
            MaskConfig(num_blocks=2, spatial_scale=0.7),  # long-range
        ]

        self._counter = Value("i", random.randrange(2**32))

    @property
    def n_tubelets(self) -> int:
        """Total number of tubelets in the video grid."""
        return self._n_t * self._n_h * self._n_w

    def _step(self) -> int:
        with self._counter.get_lock():
            self._counter.value += 1
            return self._counter.value

    def _sample_spatial_block(
        self,
        generator: torch.Generator,
        spatial_scale: float,
    ) -> tuple[int, int, int, int]:
        """Sample a rectangular spatial block.

        Args:
            generator: Random number generator.
            spatial_scale: Target spatial coverage ratio.

        Returns:
            Block coordinates as (top, bottom, left, right).
        """
        ratio_rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = self._aspect_ratio
        aspect_ratio = min_ar + ratio_rand * (max_ar - min_ar)

        target_area = spatial_scale * self._n_h * self._n_w
        h = int(min(max(round(math.sqrt(target_area / aspect_ratio)), 1), self._n_h))
        w = int(min(max(round(math.sqrt(target_area * aspect_ratio)), 1), self._n_w))

        top = int(torch.randint(max(self._n_h - h + 1, 1), (1,), generator=generator))
        left = int(torch.randint(max(self._n_w - w + 1, 1), (1,), generator=generator))

        return top, top + h, left, left + w

    def _sample_single_mask(
        self,
        generator: torch.Generator,
        config: MaskConfig,
    ) -> Tensor:
        """Sample a single spatial mask using the given configuration.

        Args:
            generator: Random number generator.
            config: Mask configuration specifying blocks and scale.

        Returns:
            Boolean spatial mask of shape [n_h, n_w]. True = masked.
        """
        mask = torch.zeros(self._n_h, self._n_w, dtype=torch.bool)
        for _ in range(config.num_blocks):
            top, bottom, left, right = self._sample_spatial_block(
                generator, config.spatial_scale
            )
            mask[top:bottom, left:right] = True
        return mask

    def _extend_to_tubes(self, spatial_mask: Tensor) -> Tensor:
        """Extend 2D spatial mask to 3D tube mask across all temporal
        positions.

        Args:
            spatial_mask: Spatial mask of shape [n_h, n_w].

        Returns:
            Flattened 3D tube mask of shape [n_t * n_h * n_w].
        """
        return spatial_mask.flatten().repeat(self._n_t)

    def sample_masks_and_target(
        self,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        """Sample encoder mask and predictor target following V-JEPA strategy.

        Args:
            generator: Random number generator.

        Returns:
            A tuple containing:
                - encoder_mask: Boolean mask for context encoder [n_tubelets].
                - predictor_target: Boolean mask for predictor target [n_tubelets].
        """
        spatial_masks = [
            self._sample_single_mask(generator, config) for config in self._mask_configs
        ]

        combined_spatial = torch.stack(spatial_masks).any(dim=0)

        # Ensure minimum unmasked patches
        unmasked_count = int((~combined_spatial).sum())
        if unmasked_count < self._min_keep:
            masked_indices = (
                combined_spatial.flatten().nonzero(as_tuple=False).squeeze(-1)
            )
            n_to_unmask = self._min_keep - unmasked_count
            perm = torch.randperm(len(masked_indices), generator=generator)[
                :n_to_unmask
            ]
            flat_mask = combined_spatial.flatten()
            flat_mask[masked_indices[perm]] = False
            combined_spatial = flat_mask.reshape(self._n_h, self._n_w)

        encoder_mask = self._extend_to_tubes(combined_spatial)

        target_idx = int(torch.randint(len(spatial_masks), (1,), generator=generator))
        predictor_target = self._extend_to_tubes(spatial_masks[target_idx])

        return encoder_mask, predictor_target

    def __call__(
        self,
        videos: Sequence[tuple[Tensor, ...] | Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Collate videos and create masks for V-JEPA training.

        Args:
            videos: List of video tensors [channels, time, height, width].

        Returns:
            A tuple containing:
                - collated_videos: Collated videos [batch, channels, T, H, W].
                - encoder_masks: Boolean masks for context encoder [batch, n_tubelets].
                - predictor_targets: Boolean masks for predictor [batch, n_tubelets].
        """
        tensor_list = [v if isinstance(v, Tensor) else v[0] for v in videos]
        collated_videos: Tensor = default_collate(tensor_list)

        seed = self._step()
        generator = torch.Generator()
        generator.manual_seed(seed)

        encoder_masks = []
        predictor_targets = []
        for _ in range(len(videos)):
            enc_mask, pred_target = self.sample_masks_and_target(generator)
            encoder_masks.append(enc_mask)
            predictor_targets.append(pred_target)

        return (
            collated_videos,
            torch.stack(encoder_masks),
            torch.stack(predictor_targets),
        )
