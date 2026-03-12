"""Evaluation entry point for trained representation models."""

import logging
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig

from exp.models.components.patchfier import VideoPatchifier
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

    models = instantiate(cfg.model)

    encoder = models["context_encoder"]
    encoder = CheckpointManager.load_encoder(
        checkpoint_path, "context_encoder", encoder
    )
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

    # --- Reconstruction Evaluation ---
    logger.info("Running reconstruction evaluation...")
    from exp.evaluation.reconstruction import ReconstructionEvaluator

    decoder = instantiate(
        cfg.evaluation.reconstruction.decoder,
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
    from exp.evaluation.prediction import PredictionEvaluator

    flat_features = features.flatten(1)  # [N, n_tubelets * embed_dim]
    feat_dim = flat_features.shape[1]

    predictor = instantiate(
        cfg.evaluation.prediction.predictor,
        input_dim=feat_dim,
        hidden_dim=feat_dim,  # match feature dim
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

    # --- Baseline ---
    logger.info("Running baseline evaluation...")
    total_features = n_tubelets[0] * n_tubelets[1] * n_tubelets[2] * cfg.model.embed_dim
    baseline = instantiate(
        cfg.evaluation.baseline,
        target_feature_size=total_features,
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

    # --- VAE Baseline ---
    logger.info("Running VAE baseline evaluation...")
    from exp.evaluation.vae_baseline import VAEBaselineEvaluator

    vae_eval = VAEBaselineEvaluator(
        in_channels=cfg.model.in_channels,
        latent_dim=cfg.evaluation.get("vae_baseline", {}).get("latent_dim", 256),
        device=device,
    )
    vae_eval.train_vae(
        video_tensor,
        num_epochs=cfg.evaluation.get("vae_baseline", {}).get("num_epochs", 50),
        batch_size=cfg.evaluation.get("vae_baseline", {}).get("batch_size", 32),
        lr=cfg.evaluation.get("vae_baseline", {}).get("learning_rate", 1e-3),
    )
    vae_result = vae_eval.evaluate(video_tensor)
    logger.info(
        f"VAE Baseline - MAE: {vae_result.recon_mae:.6f}, MSE: {vae_result.recon_mse:.6f}"
    )

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
