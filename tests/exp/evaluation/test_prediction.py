import torch

from exp.evaluation.prediction import PredictionEvaluator, PredictionResult
from exp.models.mingru import MinGRU


class TestPredictionEvaluator:
    def test_create_sequences_shape(self):
        predictor = MinGRU(input_dim=8, hidden_dim=16, output_dim=8)
        evaluator = PredictionEvaluator(predictor, horizons=[1, 2, 4])
        features = torch.randn(100, 8)
        inputs, targets = evaluator.create_sequences(features, seq_len=10)
        assert inputs.shape == (87, 10, 8)  # 100 - 10 - 4 + 1 = 87
        assert targets.shape == (87, 4, 8)  # max_horizon = 4

    def test_create_sequences_insufficient_data(self):
        predictor = MinGRU(input_dim=8, hidden_dim=16, output_dim=8)
        evaluator = PredictionEvaluator(predictor, horizons=[1, 2, 4, 8])
        features = torch.randn(5, 8)  # too few
        try:
            evaluator.create_sequences(features, seq_len=10)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_train_predictor_returns_losses(self):
        predictor = MinGRU(input_dim=8, hidden_dim=16, output_dim=8)
        evaluator = PredictionEvaluator(predictor, horizons=[1, 2])
        features = torch.randn(50, 8)
        losses = evaluator.train_predictor(
            features, seq_len=10, num_epochs=2, batch_size=8
        )
        assert len(losses) == 2

    def test_evaluate_returns_result(self):
        predictor = MinGRU(input_dim=8, hidden_dim=16, output_dim=8)
        evaluator = PredictionEvaluator(predictor, horizons=[1, 2, 4])
        features = torch.randn(50, 8)
        result = evaluator.evaluate(features, seq_len=10, batch_size=8)
        assert isinstance(result, PredictionResult)
        assert set(result.horizon_errors.keys()) == {1, 2, 4}
        assert all(v >= 0 for v in result.horizon_errors.values())
