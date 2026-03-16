"""Visualization entry point for evaluation results."""

import logging
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig

from exp.evaluation.visualization import (
    plot_prediction_errors,
    plot_reconstruction_errors,
)
from exp.oc_resolvers import register_custom_resolvers

register_custom_resolvers()
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="plot_eval")
def main(cfg: DictConfig) -> None:
    """Plot evaluation results with EMA smoothing."""
    eval_dirs = list(cfg.eval_dirs)
    labels = (
        list(cfg.labels)
        if cfg.get("labels")
        else [Path(d).parent.name for d in eval_dirs]
    )
    ema_span = cfg.ema_span
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reconstruction results
    recon_results: dict[str, dict[str, Any]] = {}
    for label, eval_dir in zip(labels, eval_dirs):
        recon_path = Path(eval_dir) / "reconstruction.pt"
        if recon_path.exists():
            recon_results[label] = torch.load(recon_path, weights_only=False)

    if recon_results:
        plot_reconstruction_errors(
            recon_results, ema_span, output_dir / "reconstruction_errors.png"
        )
        logger.info(
            f"Saved reconstruction error plot to {output_dir / 'reconstruction_errors.png'}"
        )

    # Load prediction results
    pred_results: dict[str, dict[str, Any]] = {}
    for label, eval_dir in zip(labels, eval_dirs):
        pred_path = Path(eval_dir) / "prediction.pt"
        if pred_path.exists():
            pred_results[label] = torch.load(pred_path, weights_only=False)

    if pred_results:
        # Plot for each horizon found in first result
        first_result = next(iter(pred_results.values()))
        for horizon in first_result["pointwise_horizon_errors"]:
            plot_prediction_errors(
                pred_results,
                horizon,
                ema_span,
                output_dir / f"prediction_errors_h{horizon}.png",
            )
            logger.info(f"Saved prediction error plot (horizon={horizon})")

    logger.info("Plotting complete.")


if __name__ == "__main__":
    main()
