import itertools
import math
import random
from collections.abc import Callable
from functools import partial
from multiprocessing import Value
from typing import TypedDict, override

import torch
import torch.nn.functional as F
from pamiq_core import DataUser
from pamiq_core.data.impls import RandomReplacementBuffer
from pamiq_core.torch import OptimizersSetup, get_device
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset, default_collate

from exp.buffers import BufferName
from exp.models import ModelName
from exp.models.jepa import Encoder, Predictor
from exp.utils import Size2D, size_2d_to_int_tuple

from .base import ExperimentTrainer

OPTIMIZER_NAME = "optimizer"


class JEPATrainer(ExperimentTrainer):
    """Trainer for Joint Embedding Predictive Architecture (I-JEPA).

    This trainer implements the JEPA training process which involves:
    1. A context encoder that encodes patches with some masked areas
    2. A target encoder that encodes full patches (without masking)
    3. A predictor that predicts target encoder outputs from context encoder outputs

    The training uses a masked prediction task where the model learns to predict
    representations of target patches from context patches. The target encoder
    is updated using an exponential moving average of the context encoder parameters.
    """

    @override
    def __init__(
        self,
        partial_optimizer: partial[Optimizer],
        collate_fn: Callable[[list[tuple[Tensor]]], tuple[Tensor, Tensor, Tensor]],
        context_encoder_name: str = ModelName.JEPA_CONTEXT_ENCODER,
        target_encoder_name: str = ModelName.JEPA_TARGET_ENCODER,
        predictor_name: str = ModelName.JEPA_PREDICTOR,
        data_user_name: str = BufferName.IMAGE,
        log_prefix: str = "jepa",
        target_encoder_update_moving_average: float = 0.996,  # based on the original I-JEPA initinal setting.
        batch_size: int = 1,
        max_epochs: int = 1,
        min_buffer_size: int = 0,
        min_new_data_count: int = 0,
        max_steps_every_train: int | None = None,
    ) -> None:
        """Initialize the JEPA trainer.

        Args:
            partial_optimizer: Partially configured optimizer to be used with
                the model parameters.
            context_encoder_name: Name of the context encoder model to retrieve
                from the model registry.
            target_encoder_name: Name of the target encoder model to retrieve
                from the model registry.
            predictor_name: Name of the predictor model to retrieve from the
                model registry.
            data_user_name: Name of the data user providing training data.
            collate_fn: Collator function for sampling input data, encoder mask and predictor target.
            log_prefix: Prefix for training metrics in Aim logging.
            target_encoder_update_moving_average: Momentum coefficient for updating
                the target encoder from the context encoder (higher values mean
                slower updates, default: 0.996 based on original I-JEPA).
            batch_size: Data sample size for 1 step.
            max_epochs: Maximum number of epochs to train per training session.
            min_buffer_size: Minimum buffer size required before training starts.
            min_new_data_count: Minimum number of new data points required for training.
            max_steps_every_train: Maximum number of steps to train per session.
                If set, training stops after this many steps.
        """
        super().__init__(data_user_name, min_buffer_size, min_new_data_count)

        self.data_user_name = data_user_name
        self.partial_optimizer = partial_optimizer
        self.partial_dataloader = partial(
            DataLoader,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        self.context_encoder_name = context_encoder_name
        self.target_encoder_name = target_encoder_name
        self.predictor_name = predictor_name
        self.log_prefix = log_prefix
        self.target_encoder_update_moving_average = target_encoder_update_moving_average
        self.max_epochs = max_epochs
        self.max_steps_every_train = max_steps_every_train

    @override
    def on_data_users_attached(self) -> None:
        """Set up data user references when they are attached to the trainer.

        This method is called automatically by the PAMIQ framework when
        data users are attached to the trainer. It retrieves and stores
        references to the required data users for convenient access
        during training.
        """
        super().on_data_users_attached()
        self.data_user: DataUser[list[Tensor]] = self.get_data_user(self.data_user_name)

    @override
    def on_training_models_attached(self) -> None:
        """Set up model references when they are attached to the trainer.

        This method is called automatically by the PAMIQ framework when
        training models are attached to the trainer. It retrieves and
        stores references to the JEPA models (context encoder, target
        encoder, and predictor) for convenient access during training.
        """
        super().on_training_models_attached()
        self.context_encoder = self.get_torch_training_model(
            self.context_encoder_name, Encoder
        )
        self.target_encoder = self.get_torch_training_model(
            self.target_encoder_name, Encoder
        )
        self.predictor = self.get_torch_training_model(self.predictor_name, Predictor)

    @override
    def create_optimizers(self) -> OptimizersSetup:
        """Create optimizers for JEPA training.

        Creates an optimizer that updates both the context encoder and predictor
        parameters. The target encoder is updated separately through exponential
        moving average and is not directly optimized.

        Returns:
            Dictionary mapping optimizer name to configured optimizer instance.
        """
        return {
            OPTIMIZER_NAME: self.partial_optimizer(
                itertools.chain(
                    self.context_encoder.model.parameters(),
                    self.predictor.model.parameters(),
                )
            )
        }

    @override
    def setup(self) -> None:
        super().setup()
        self.device = get_device(self.context_encoder.model)

    class LossDict(TypedDict):
        """Dictionary structure for JEPA loss computation results."""

        loss: Tensor
        loss_per_data: Tensor
        latent_context: Tensor
        latent_target: Tensor
        latent_predictor: Tensor

    @classmethod
    def compute_loss(
        cls,
        target_encoder: Encoder,
        context_encoder: Encoder,
        predictor: Predictor,
        batch: tuple[Tensor, Tensor, Tensor],
    ) -> LossDict:
        """Compute JEPA loss between predicted and target representations.

        Args:
            target_encoder: Target encoder model (used without gradients).
            context_encoder: Context encoder model that processes masked inputs.
            predictor: Predictor model that predicts target representations.
            batch: Tuple containing:
                - data: Input images (shape: [batch_size, channels, height, width])
                - masks_for_context_encoder: Boolean masks for context encoder
                  (shape: [batch_size, n_patches], True = masked)
                - targets_for_predictor: Boolean masks indicating which patches
                  to predict (shape: [batch_size, n_patches], True = predict)

        Returns:
            LossDict containing:
                - loss: Averaged smooth L1 loss over predicted patches
                - latent_context: Context encoder representations
                - latent_target: Target encoder representations (normalized)
                - latent_predictor: Predictor output representations
        """
        (data, masks_for_context_encoder, targets_for_predictor) = batch

        # target encoder
        with torch.no_grad():
            latent_target = target_encoder(data)
            # normalize over feature-dim
            latent_target = F.layer_norm(
                latent_target,
                (latent_target.size(-1),),
            )

        # context encoder
        latent_context = context_encoder(data, masks_for_context_encoder)

        # predictor
        latent_predictor = predictor(latent_context, targets_for_predictor)

        # Element wise smooth l1 loss for masking.
        losses = F.smooth_l1_loss(
            latent_predictor, latent_target, reduction="none"
        ).mean(-1)
        # shape: [batch, n_patches]

        # Ignore patches that are not selected for prediction.
        losses = torch.masked_fill(losses, ~targets_for_predictor, 0.0)
        loss_per_data = losses.sum(-1) / targets_for_predictor.sum(-1)
        loss = losses.sum() / targets_for_predictor.sum()

        return cls.LossDict(
            loss=loss,
            latent_context=latent_context,
            latent_target=latent_target,
            latent_predictor=latent_predictor,
            loss_per_data=loss_per_data,
        )

    @override
    def create_dataloader(self) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
        dataset = TensorDataset(torch.stack(self.data_user.get_data()))
        return self.partial_dataloader(dataset=dataset)

    @override
    def batch_step(self, batch: tuple[Tensor, Tensor, Tensor], index: int) -> None:
        batch = tuple(map(lambda x: x.to(self.device), batch))  # pyright: ignore [reportAssignmentType, ]
        self.optimizers[OPTIMIZER_NAME].zero_grad()
        loss_dict = self.compute_loss(
            self.target_encoder.model,
            self.context_encoder.model,
            self.predictor.model,
            batch,
        )

        loss = loss_dict["loss"]
        loss.backward()
        self.optimizers[OPTIMIZER_NAME].step()

        # target_encoder updates weights by moving average from context_encoder
        with torch.no_grad():
            # In the original I-JEPA, m changes through training process.
            # But in ami-q, since assuming Semi-permanent training, m is set as fixed value.
            m = self.target_encoder_update_moving_average
            for target_encoder_param, context_encoder_param in zip(
                self.target_encoder.model.parameters(),
                self.context_encoder.model.parameters(),
                strict=True,
            ):
                target_encoder_param.data.mul_(m).add_(
                    (1.0 - m) * context_encoder_param.detach().data
                )
        metrics = {
            "target_encoder_latent_std": loss_dict["latent_target"]
            .std(0)
            .mean()
            .item(),
            "context_encoder_latent_std": loss_dict["latent_context"]
            .std(0)
            .mean()
            .item(),
            "loss": loss.item(),
        }

        for tag, value in metrics.items():
            self.aim_run.track(
                value,
                name=tag,
                step=self.global_steps,
                context=self.default_aim_context,
            )

    @staticmethod
    def create_buffer(
        batch_size: int,
        iteration_count: int,
        expected_survival_length: int,
    ) -> RandomReplacementBuffer[Tensor]:
        """Create data buffer for this trainer."""
        return RandomReplacementBuffer[Tensor](
            max_size=iteration_count * batch_size,
            expected_survival_length=expected_survival_length,
        )


class MultiBlockMaskCollator2d:
    """JEPA collator function for providing boolean mask tensors.

    This collator creates boolean masks for both the context encoder and predictor target.
    It's designed to work with the Image-JEPA (Image Joint Embedding Predictive Architecture) model.

    The masks are boolean tensors where:
    - True values indicate patches to be masked (ignored)
    - False values indicate patches to be processed or predicted
    """

    def __init__(
        self,
        num_patches: Size2D,
        mask_scale: tuple[float, float] = (0.10, 0.25),
        n_masks: int = 4,
        aspect_ratio: tuple[float, float] = (0.75, 1.5),
        min_keep: int = 10,
    ) -> None:
        """Initialize the MultiBlockMaskCollator.

        Args:
            input_size: Size of the input image.
            num_patches: Height and width for patched image.
            mask_scale: Range of mask scale (min, max).
            n_masks: Number of mask candidates to generate.
            aspect_ratio: Range of aspect ratios for masks.
            min_keep: Minimum number of patches to keep unmasked.
        """
        super().__init__()
        if mask_scale[0] > mask_scale[1]:
            raise ValueError("mask_scale[0] must be less than mask_scale[1]")
        if mask_scale[0] < 0:
            raise ValueError("mask_scale[0] must be greater than 0")
        if mask_scale[1] > 1:
            raise ValueError("mask_scale[1] must be less than 1")

        num_patches = size_2d_to_int_tuple(num_patches)

        self.n_patches_height = num_patches[0]
        self.n_patches_width = num_patches[1]

        if min_keep > self.n_patches:
            raise ValueError(
                f"min_keep ({min_keep}) must be less than or equal to total patches "
                f"({self.n_patches_height * self.n_patches_width})"
            )

        self.mask_scale = mask_scale
        self.n_masks = n_masks
        self.aspect_ratio = aspect_ratio
        self.min_keep = min_keep
        self._itr_counter = Value("i", random.randrange(2**32))

    @property
    def n_patches(self) -> int:
        """Total number of patches in the image grid."""
        return self.n_patches_height * self.n_patches_width

    def step(self) -> int:
        """Increment and return the iteration counter."""
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_mask_rectangle(
        self,
        generator: torch.Generator,
    ) -> tuple[int, int, int, int]:
        """Randomly sample a rectangular mask.

        Args:
            generator: Generator for pseudo-random numbers.

        Returns:
            Top, bottom, left, and right coordinates of the mask.
        """
        _scale_rand, _ratio_rand = torch.rand(2, generator=generator).tolist()
        # -- Sample mask scale
        min_s, max_s = self.mask_scale
        mask_scale = min_s + _scale_rand * (max_s - min_s)
        max_keep = mask_scale * self.n_patches

        # -- Sample mask aspect-ratio
        min_ar, max_ar = self.aspect_ratio
        aspect_ratio = min_ar + _ratio_rand * (max_ar - min_ar)

        # -- Compute height and width of mask (given scale and aspect-ratio)
        patch_ar = self.n_patches_width / self.n_patches_height
        if patch_ar > aspect_ratio:
            h_max = self.n_patches_height
            w_max = self.n_patches_height * aspect_ratio
        else:
            h_max = self.n_patches_width / aspect_ratio
            w_max = self.n_patches_width

        num_patches_max = h_max * w_max
        scale = math.sqrt(max_keep / num_patches_max)
        h, w = round(scale * h_max), round(scale * w_max)

        # Apply min keep
        if h * w < self.min_keep:
            scale = math.sqrt(self.min_keep / num_patches_max)
            h, w = math.ceil(scale * h_max), math.ceil(scale * w_max)

        # clamp
        h = min(max(h, 1), self.n_patches_height)
        w = min(max(w, 1), self.n_patches_width)

        # -- Compute mask coordinates
        top = int(
            torch.randint(
                high=self.n_patches_height - h + 1, size=(1,), generator=generator
            ).item()
        )
        left = int(
            torch.randint(
                high=self.n_patches_width - w + 1, size=(1,), generator=generator
            ).item()
        )
        bottom = top + h
        right = left + w

        return top, bottom, left, right

    def sample_masks_and_target(
        self, generator: torch.Generator
    ) -> tuple[Tensor, Tensor]:
        """Sample boolean masks for the encoder and a target mask for the
        predictor.

        Args:
            generator: Generator for pseudo-random numbers.

        Returns:
            A tuple containing:
                - encoder_mask: Boolean mask for the encoder (True for masked patches)
                - predictor_target: Boolean mask representing the target for the predictor
        """
        sampled_masks = []
        for _ in range(self.n_masks):
            mask = torch.zeros(
                self.n_patches_height, self.n_patches_width, dtype=torch.bool
            )
            top, bottom, left, right = self._sample_mask_rectangle(generator)
            mask[top:bottom, left:right] = True
            sampled_masks.append(mask.flatten())

        # Create encoder mask by combining all sampled masks
        encoder_mask = torch.stack(sampled_masks).sum(0).type(torch.bool)

        # Randomly select one mask as the predictor target
        mask_idx = int(
            torch.randint(
                high=len(sampled_masks), size=(1,), generator=generator
            ).item()
        )
        predictor_target = sampled_masks[mask_idx]

        # Apply min keep
        if encoder_mask.logical_not().sum() < self.min_keep:
            indices = torch.randperm(self.n_patches, generator=generator)[
                : self.min_keep
            ]
            encoder_mask[indices] = False
            predictor_target[indices] = True

        return encoder_mask, predictor_target

    def __call__(
        self, images: list[tuple[Tensor] | Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Collate input images and create boolean masks for context encoder
        and predictor target.

        Args:
            images: List of image tensors. Each image is shape [3, height, width].

        Returns:
            A tuple containing:
                - collated_images: Collated images (shape: [batch_size, 3, height, width])
                - collated_encoder_masks: Boolean masks for context encoder (shape: [batch_size, n_patches])
                - collated_predictor_targets: Boolean masks representing predictor targets
                  (shape: [batch_size, n_patches])
        """
        tensor_list = [im if isinstance(im, Tensor) else im[0] for im in images]
        collated_images: Tensor = default_collate(tensor_list)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        encoder_masks, predictor_targets = [], []
        for _ in range(len(images)):
            enc_mask, pred_target = self.sample_masks_and_target(g)
            encoder_masks.append(enc_mask)
            predictor_targets.append(pred_target)

        collated_encoder_masks = torch.stack(encoder_masks)
        collated_predictor_targets = torch.stack(predictor_targets)

        return (
            collated_images,
            collated_encoder_masks,
            collated_predictor_targets,
        )
