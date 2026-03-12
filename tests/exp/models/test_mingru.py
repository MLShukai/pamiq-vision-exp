import torch

from exp.models.mingru import MinGRU, MinGRUCell


class TestMinGRUCell:
    def test_output_shape(self):
        cell = MinGRUCell(16, 32)
        x = torch.randn(4, 16)
        h = torch.zeros(4, 32)
        h_new = cell(x, h)
        assert h_new.shape == (4, 32)

    def test_gradient_flows(self):
        cell = MinGRUCell(16, 32)
        x = torch.randn(4, 16, requires_grad=True)
        h = torch.zeros(4, 32)
        h_new = cell(x, h)
        h_new.sum().backward()
        assert x.grad is not None


class TestMinGRU:
    def test_output_shape(self):
        model = MinGRU(input_dim=16, hidden_dim=32, output_dim=16)
        x = torch.randn(4, 10, 16)  # batch=4, seq_len=10
        output, hidden = model(x)
        assert output.shape == (4, 10, 16)
        assert len(hidden) == 1
        assert hidden[0].shape == (4, 32)

    def test_multi_layer(self):
        model = MinGRU(input_dim=16, hidden_dim=32, output_dim=16, num_layers=3)
        x = torch.randn(4, 10, 16)
        output, hidden = model(x)
        assert output.shape == (4, 10, 16)
        assert len(hidden) == 3

    def test_with_initial_hidden(self):
        model = MinGRU(input_dim=16, hidden_dim=32, output_dim=16)
        x = torch.randn(4, 10, 16)
        h0 = [torch.randn(4, 32)]
        output, hidden = model(x, h0)
        assert output.shape == (4, 10, 16)

    def test_gradient_flows(self):
        model = MinGRU(input_dim=16, hidden_dim=32, output_dim=16)
        x = torch.randn(4, 10, 16, requires_grad=True)
        output, _ = model(x)
        output.sum().backward()
        assert x.grad is not None
