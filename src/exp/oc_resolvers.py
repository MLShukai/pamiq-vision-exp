"""OmegaConf custom resolvers."""

import math

from omegaconf import OmegaConf

_registered = False


def compute_feature_size(
    video_shape: list[int], tubelet_size: list[int], embed_dim: int
) -> int:
    """Compute feature_size = n_tubelets_total * embed_dim."""
    n_t = video_shape[0] // tubelet_size[0]
    n_h = video_shape[1] // tubelet_size[1]
    n_w = video_shape[2] // tubelet_size[2]
    return n_t * n_h * n_w * embed_dim


def register_custom_resolvers() -> None:
    """Register custom OmegaConf resolvers.

    Safe to call multiple times.
    """
    global _registered
    if not _registered:
        OmegaConf.register_new_resolver("feature_size", compute_feature_size)
        _registered = True
