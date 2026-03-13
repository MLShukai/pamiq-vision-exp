"""OmegaConf custom resolvers."""

from omegaconf import OmegaConf

_registered = False


def compute_feature_size(
    video_shape: list[int], tubelet_size: list[int], embed_dim: int
) -> int:
    """Compute the total feature size produced by a tubelet-based encoder.

    The feature size equals the number of tubelets (T * H * W) times the
    embedding dimension, where the tubelet count along each axis is
    ``video_shape[i] // tubelet_size[i]``.

    Args:
        video_shape: Video dimensions as ``[T, H, W]``.
        tubelet_size: Tubelet dimensions as ``[t, h, w]``.
        embed_dim: Embedding dimension per tubelet.

    Returns:
        Total feature size (n_tubelets * embed_dim).
    """
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
