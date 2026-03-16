from pathlib import Path

import pytest
import torch

from exp.evaluation.visualization import (
    compute_ema,
    plot_prediction_errors,
    plot_reconstruction_errors,
)


class TestComputeEma:
    def test_single_value(self):
        values = torch.tensor([5.0])
        result = compute_ema(values, span=10.0)
        assert result.shape == (1,)
        assert result[0].item() == pytest.approx(5.0)

    def test_constant_input(self):
        values = torch.full((20,), 3.0)
        result = compute_ema(values, span=5.0)
        # EMA of constant signal should converge to that constant
        assert result[-1].item() == pytest.approx(3.0)
        assert result[0].item() == pytest.approx(3.0)

    def test_known_sequence(self):
        values = torch.tensor([1.0, 2.0, 3.0])
        span = 3.0
        # alpha = 2/(3+1) = 0.5
        # ema[0] = 1.0
        # ema[1] = 0.5 * 2.0 + 0.5 * 1.0 = 1.5
        # ema[2] = 0.5 * 3.0 + 0.5 * 1.5 = 2.25
        result = compute_ema(values, span)
        assert result[0].item() == pytest.approx(1.0)
        assert result[1].item() == pytest.approx(1.5)
        assert result[2].item() == pytest.approx(2.25)

    def test_output_shape_matches_input(self):
        values = torch.randn(100)
        result = compute_ema(values, span=10.0)
        assert result.shape == values.shape

    @pytest.mark.parametrize("span", [2.0, 10.0, 100.0])
    def test_first_element_equals_input(self, span: float):
        values = torch.randn(50)
        result = compute_ema(values, span)
        assert result[0].item() == pytest.approx(values[0].item())


class TestPlotReconstructionErrors:
    def test_saves_figure(self, tmp_path: Path):
        results = {
            "model_a": {"pointwise_mae": torch.rand(50)},
            "model_b": {"pointwise_mae": torch.rand(50)},
        }
        output_path = tmp_path / "recon.png"
        plot_reconstruction_errors(results, ema_span=5.0, output_path=output_path)
        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestPlotPredictionErrors:
    def test_saves_figure(self, tmp_path: Path):
        results = {
            "model_a": {
                "pointwise_horizon_errors": {1: torch.rand(30), 4: torch.rand(30)}
            },
        }
        output_path = tmp_path / "pred.png"
        plot_prediction_errors(
            results, horizon=1, ema_span=5.0, output_path=output_path
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 0
