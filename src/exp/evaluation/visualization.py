"""Visualization utilities for evaluation results."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from torch import Tensor


def compute_ema(values: Tensor, span: float) -> Tensor:
    """Compute forward exponential moving average (past to future).

    Args:
        values: 1D tensor of values.
        span: EMA span parameter.

    Returns:
        1D tensor of EMA-smoothed values, same length as input.
    """
    alpha = 2.0 / (span + 1.0)
    ema = values.clone()
    for t in range(1, len(values)):
        ema[t] = alpha * values[t] + (1.0 - alpha) * ema[t - 1]
    return ema


def plot_reconstruction_errors(
    results: dict[str, dict[str, Any]],
    ema_span: float,
    output_path: Path,
) -> None:
    """Plot per-point reconstruction MAE with EMA smoothing.

    Args:
        results: Mapping from label to a loaded .pt dict with ``pointwise_mae`` key.
        ema_span: EMA span for smoothing.
        output_path: Path to save the figure.
    """
    fig, ax = plt.subplots()
    for label, data in results.items():
        raw = data["pointwise_mae"].float()
        smoothed = compute_ema(raw, ema_span)
        ax.plot(raw.numpy(), alpha=0.3, linewidth=0.5, label=f"{label} (raw)")
        ax.plot(smoothed.numpy(), linewidth=2.0, label=f"{label} (EMA)")
    ax.set_xlabel("Data point index")
    ax.set_ylabel("MAE")
    ax.set_title("Reconstruction Error")
    ax.legend()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_errors(
    results: dict[str, dict[str, Any]],
    horizon: int,
    ema_span: float,
    output_path: Path,
) -> None:
    """Plot per-sequence prediction MAE for a given horizon with EMA smoothing.

    Args:
        results: Mapping from label to a loaded .pt dict with
            ``pointwise_horizon_errors`` key.
        horizon: Prediction horizon to plot.
        ema_span: EMA span for smoothing.
        output_path: Path to save the figure.
    """
    fig, ax = plt.subplots()
    for label, data in results.items():
        raw = data["pointwise_horizon_errors"][horizon].float()
        smoothed = compute_ema(raw, ema_span)
        ax.plot(raw.numpy(), alpha=0.3, linewidth=0.5, label=f"{label} (raw)")
        ax.plot(smoothed.numpy(), linewidth=2.0, label=f"{label} (EMA)")
    ax.set_xlabel("Sequence index")
    ax.set_ylabel("MAE")
    ax.set_title(f"Prediction Error (horizon={horizon})")
    ax.legend()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
