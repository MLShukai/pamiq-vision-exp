"""Evaluation entry point for trained representation models."""

import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from exp.evaluation.prediction import PredictionEvaluator
from exp.evaluation.reconstruction import ReconstructionEvaluator
from exp.oc_resolvers import register_custom_resolvers
from exp.training.checkpoint import CheckpointManager

register_custom_resolvers()

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
        feat_dim = features.shape[1]

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
            features,
            seq_len=cfg.evaluation.prediction.seq_len,
            num_epochs=cfg.evaluation.prediction.num_epochs,
            batch_size=cfg.evaluation.prediction.batch_size,
            lr=cfg.evaluation.prediction.learning_rate,
        )
        pred_result = pred_evaluator.evaluate(
            features, seq_len=cfg.evaluation.prediction.seq_len
        )
        for horizon, error in pred_result.horizon_errors.items():
            logger.info(f"Prediction horizon {horizon} - MAE: {error:.6f}")

    # --- Downsampling Baseline ---
    if cfg.evaluation.get("baseline") is not None:
        logger.info("Running downsampling baseline...")
        baseline = instantiate(cfg.evaluation.baseline).to(device)

        # Encode with baseline (same loop as main encoder)
        baseline_features_list = []
        with torch.no_grad():
            for i in range(0, len(video_tensor), batch_size):
                b = video_tensor[i : i + batch_size].to(device)
                feat = baseline(b)
                baseline_features_list.append(feat.cpu())
        baseline_features = torch.cat(baseline_features_list, dim=0)
        logger.info(f"Baseline features shape: {baseline_features.shape}")

        # Run reconstruction evaluation on baseline features
        if cfg.evaluation.get("reconstruction") is not None:
            baseline_decoder = instantiate(
                cfg.evaluation.reconstruction.decoder,
                _convert_="partial",
            ).to(device)
            baseline_recon_eval = ReconstructionEvaluator(
                baseline, baseline_decoder, device
            )
            baseline_recon_eval.train_decoder(
                baseline_features,
                video_tensor,
                num_epochs=cfg.evaluation.reconstruction.num_epochs,
                batch_size=cfg.evaluation.reconstruction.batch_size,
                lr=cfg.evaluation.reconstruction.learning_rate,
            )
            baseline_recon_result = baseline_recon_eval.evaluate(
                baseline_features, video_tensor
            )
            logger.info(
                f"Baseline Reconstruction - MAE: {baseline_recon_result.mae:.6f}, "
                f"MSE: {baseline_recon_result.mse:.6f}"
            )

        # Run prediction evaluation on baseline features
        if cfg.evaluation.get("prediction") is not None:
            baseline_feat_dim = baseline_features.shape[1]
            baseline_predictor = instantiate(
                cfg.evaluation.prediction.predictor,
                input_dim=baseline_feat_dim,
                hidden_dim=baseline_feat_dim,
                output_dim=baseline_feat_dim,
            ).to(device)
            baseline_pred_eval = PredictionEvaluator(
                predictor=baseline_predictor,
                horizons=list(cfg.evaluation.prediction.horizons),
                device=device,
            )
            baseline_pred_eval.train_predictor(
                baseline_features,
                seq_len=cfg.evaluation.prediction.seq_len,
                num_epochs=cfg.evaluation.prediction.num_epochs,
                batch_size=cfg.evaluation.prediction.batch_size,
                lr=cfg.evaluation.prediction.learning_rate,
            )
            baseline_pred_result = baseline_pred_eval.evaluate(
                baseline_features, seq_len=cfg.evaluation.prediction.seq_len
            )
            for horizon, error in baseline_pred_result.horizon_errors.items():
                logger.info(f"Baseline Prediction horizon {horizon} - MAE: {error:.6f}")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
