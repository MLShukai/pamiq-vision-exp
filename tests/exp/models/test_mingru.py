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

    def test_gate_depends_only_on_input(self):
        """Gate z must depend only on x, not on h.

        For minGRU: out = (1-z)*h + z*h_tilde, where z and h_tilde depend
        only on x. So (out1 - out2) = (1-z) * (h1 - h2), meaning the
        per-element ratio (out1 - out2) / (h1 - h2) equals (1-z) ∈ (0, 1).
        """
        cell = MinGRUCell(4, 8)
        x = torch.randn(1, 4)
        h1 = torch.randn(1, 8)
        h2 = torch.randn(1, 8) + 5.0  # ensure h1 != h2

        out1 = cell(x, h1)
        out2 = cell(x, h2)

        ratio = (out1 - out2) / (h1 - h2)
        # Each element is (1-z_i) which must be in (0, 1)
        assert (ratio > 0).all() and (ratio < 1).all()


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
        output_with_h0, hidden = model(x, h0)
        output_without_h0, _ = model(x)
        assert output_with_h0.shape == (4, 10, 16)
        assert len(hidden) == 1
        assert hidden[0].shape == (4, 32)
        assert not torch.allclose(output_with_h0, output_without_h0)

    def test_gradient_flows(self):
        model = MinGRU(input_dim=16, hidden_dim=32, output_dim=16)
        x = torch.randn(4, 10, 16, requires_grad=True)
        output, _ = model(x)
        output.sum().backward()
        assert x.grad is not None

    def test_parallel_matches_sequential(self):
        """Parallel scan must match step-by-step recurrence."""
        torch.manual_seed(42)
        cell = MinGRUCell(16, 32)
        cell.eval()

        batch, seq_len = 4, 10
        x = torch.randn(batch, seq_len, 16)
        h0 = torch.randn(batch, 32)

        # Parallel
        with torch.no_grad():
            h_par, h_par_final = cell.parallel(x, h0)

        # Sequential
        h_seq_list = []
        h = h0
        with torch.no_grad():
            for t in range(seq_len):
                h = cell(x[:, t], h)
                h_seq_list.append(h)
        h_seq = torch.stack(h_seq_list, dim=1)

        assert torch.allclose(h_par, h_seq, atol=1e-5)
        assert torch.allclose(h_par_final, h_seq_list[-1], atol=1e-5)

    def test_hidden_state_continuity(self):
        """Processing in two halves with hidden state must match full
        sequence."""
        torch.manual_seed(42)
        model = MinGRU(input_dim=16, hidden_dim=32, output_dim=16)
        model.eval()

        batch, seq_len = 4, 10
        x = torch.randn(batch, seq_len, 16)
        h0 = [torch.randn(batch, 32)]

        with torch.no_grad():
            # Full sequence
            out_full, _ = model(x, h0)

            # Split in two halves
            x1, x2 = x[:, :5], x[:, 5:]
            out1, hidden_mid = model(x1, h0)
            out2, _ = model(x2, hidden_mid)

            out_concat = torch.cat([out1, out2], dim=1)

        assert torch.allclose(out_full, out_concat, atol=1e-5)
