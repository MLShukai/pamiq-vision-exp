"""Evaluation entry point for trained representation models."""

import logging
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig

from exp.evaluation.baseline import DownsamplingBaseline
from exp.evaluation.prediction import PredictionEvaluator
from exp.evaluation.reconstruction import ReconstructionEvaluator
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

    # Create models and load encoder from checkpoint
    models = instantiate(cfg.model)
    encoder_key = cfg.get("encoder_key", "context_encoder")
    encoder = models[encoder_key]
    encoder = CheckpointManager.load_encoder(checkpoint_path, encoder_key, encoder)
    encoder = encoder.to(device)

    # Create data pipeline via instantiate
    frame_loader = instantiate(cfg.data.frame_loader)
    frame_stacker = instantiate(cfg.data.frame_stacker)

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

    # --- Encode dataset ---
    encoder.eval()
    features_list = []
    batch_size = cfg.evaluation.encoding_batch_size
    with torch.no_grad():
        for i in range(0, len(video_tensor), batch_size):
            batch = video_tensor[i : i + batch_size].to(device)
            feat = encoder(batch)
            features_list.append(feat.cpu())
    features = torch.cat(features_list, dim=0)  # [N, *]
    logger.info(f"Encoded features shape: {features.shape}")

    # --- Reconstruction Evaluation ---
    if cfg.evaluation.get("reconstruction") is not None:
        logger.info("Running reconstruction evaluation...")
        decoder = instantiate(
            cfg.evaluation.reconstruction.decoder,
            _convert_="partial",
        ).to(device)

        recon_evaluator = ReconstructionEvaluator(encoder, decoder, device)
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
    if cfg.evaluation.get("prediction") is not None:
        logger.info("Running prediction evaluation...")
        flat_features = features.flatten(1)  # [N, feat_dim]
        feat_dim = flat_features.shape[1]

        predictor = instantiate(
            cfg.evaluation.prediction.predictor,
            input_dim=feat_dim,
            hidden_dim=feat_dim,
            output_dim=feat_dim,
        ).to(device)

        pred_evaluator = PredictionEvaluator(
            predictor=predictor,
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

    # --- Downsampling Baseline ---
    if cfg.evaluation.get("baseline") is not None:
        logger.info("Running downsampling baseline...")
        flat_feat_size = features.flatten(1).shape[1]
        baseline = DownsamplingBaseline(
            target_feature_size=flat_feat_size,
            in_channels=cfg.model.in_channels,
        )
        baseline_features = baseline.downsample(video_tensor)
        baseline_recon = baseline.reconstruct(
            baseline_features, target_size=tuple(cfg.data.frame_loader.target_size)
        )
        target_frames = video_tensor.mean(dim=2)
        baseline_mae = F.l1_loss(baseline_recon, target_frames).item()
        baseline_mse = F.mse_loss(baseline_recon, target_frames).item()
        logger.info(f"Baseline - MAE: {baseline_mae:.6f}, MSE: {baseline_mse:.6f}")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
