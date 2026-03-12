"""Evaluation entry point for trained representation models."""

import logging
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from exp.data.loader import VideoFrameLoader
from exp.data.stacking import FrameStacker
from exp.evaluation.baseline import DownsamplingBaseline
from exp.evaluation.prediction import PredictionEvaluator
from exp.evaluation.reconstruction import ReconstructionEvaluator
from exp.models.components.patchfier import VideoPatchifier
from exp.models.mingru import MinGRU
from exp.models.vjepa import LightWeightDecoder, create_video_jepa
from exp.training.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    """Run evaluation pipeline."""
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    if cfg.checkpoint_path is None:
        raise ValueError("checkpoint_path must be specified")

    checkpoint_path = Path(cfg.checkpoint_path)
    logger.info(f"Evaluating checkpoint: {checkpoint_path}")

    # Create and load encoder
    video_shape = tuple(cfg.model.video_shape)
    tubelet_size = tuple(cfg.model.tubelet_size)
    n_tubelets = VideoPatchifier.compute_num_tubelets(video_shape, tubelet_size)

    models = create_video_jepa(
        video_shape=video_shape,
        tubelet_size=tubelet_size,
        in_channels=cfg.model.in_channels,
        hidden_dim=cfg.model.hidden_dim,
        embed_dim=cfg.model.embed_dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
    )

    encoder = models["context_encoder"]
    encoder = CheckpointManager.load_encoder(
        checkpoint_path, "context_encoder", encoder
    )
    encoder = encoder.to(device)

    # Create data pipeline for evaluation
    video_paths = [Path(p) for p in cfg.data.video_paths]
    frame_loader = VideoFrameLoader(
        video_paths=video_paths,
        target_fps=cfg.data.target_fps,
        target_size=tuple(cfg.data.target_size),
        mean=tuple(cfg.data.mean),
        std=tuple(cfg.data.std),
        fade_duration=cfg.data.fade_duration,
    )
    frame_stacker = FrameStacker(
        num_frames=cfg.data.num_frames,
        stride=cfg.data.stride,
    )

    # Collect all stacked videos
    logger.info("Collecting video frames...")
    videos = []
    for frame in frame_loader:
        stacked = frame_stacker.push(frame)
        if stacked is not None:
            videos.append(stacked)

    if not videos:
        logger.error("No video frames collected")
        return

    video_tensor = torch.stack(videos)  # [N, C, T, H, W]
    logger.info(f"Collected {len(videos)} stacked video samples")

    # --- Reconstruction Evaluation ---
    logger.info("Running reconstruction evaluation...")
    decoder = LightWeightDecoder(
        n_tubelets=n_tubelets,
        tubelet_size=tubelet_size,
        embed_dim=cfg.model.embed_dim,
    ).to(device)

    recon_evaluator = ReconstructionEvaluator(encoder, decoder, device)
    features = recon_evaluator.encode_dataset(
        video_tensor, batch_size=cfg.evaluation.reconstruction.batch_size
    )
    recon_evaluator.train_decoder(
        features,
        video_tensor,
        num_epochs=cfg.evaluation.reconstruction.num_epochs,
        batch_size=cfg.evaluation.reconstruction.batch_size,
        lr=cfg.evaluation.reconstruction.learning_rate,
    )
    recon_result = recon_evaluator.evaluate(features, video_tensor)
    logger.info(
        f"Reconstruction - MAE: {recon_result.mae:.6f}, MSE: {recon_result.mse:.6f}"
    )

    # --- Future Prediction Evaluation ---
    logger.info("Running prediction evaluation...")
    # Flatten features for sequence prediction
    flat_features = features.flatten(1)  # [N, n_tubelets * embed_dim]
    feat_dim = flat_features.shape[1]

    mingru = MinGRU(
        input_dim=feat_dim,
        hidden_dim=cfg.evaluation.prediction.hidden_dim,
        output_dim=feat_dim,
        num_layers=cfg.evaluation.prediction.num_layers,
    ).to(device)

    pred_evaluator = PredictionEvaluator(
        predictor=mingru,
        horizons=list(cfg.evaluation.prediction.horizons),
        device=device,
    )
    pred_evaluator.train_predictor(
        flat_features,
        seq_len=cfg.evaluation.prediction.seq_len,
        num_epochs=cfg.evaluation.prediction.num_epochs,
        batch_size=cfg.evaluation.prediction.batch_size,
        lr=cfg.evaluation.prediction.learning_rate,
    )
    pred_result = pred_evaluator.evaluate(
        flat_features, seq_len=cfg.evaluation.prediction.seq_len
    )
    for horizon, error in pred_result.horizon_errors.items():
        logger.info(f"Prediction horizon {horizon} - MAE: {error:.6f}")

    # --- Baseline ---
    logger.info("Running baseline evaluation...")
    total_features = n_tubelets[0] * n_tubelets[1] * n_tubelets[2] * cfg.model.embed_dim
    baseline = DownsamplingBaseline(
        target_feature_size=total_features,
        in_channels=cfg.model.in_channels,
    )
    baseline_features = baseline.downsample(video_tensor)
    baseline_recon = baseline.reconstruct(
        baseline_features, target_size=tuple(cfg.data.target_size)
    )
    # Average over time for comparison
    target_frames = video_tensor.mean(dim=2)
    baseline_mae = F.l1_loss(baseline_recon, target_frames).item()
    baseline_mse = F.mse_loss(baseline_recon, target_frames).item()
    logger.info(f"Baseline - MAE: {baseline_mae:.6f}, MSE: {baseline_mse:.6f}")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
