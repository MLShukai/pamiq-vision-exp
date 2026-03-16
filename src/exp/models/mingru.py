from typing import cast, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _parallel_scan_log(log_a: Tensor, b: Tensor, h0: Tensor | None = None) -> Tensor:
    """Parallel scan in log-space for linear recurrence h_t = a_t * h_{t-1} +
    b_t.

    Implements Equation 5 from "Were RNNs All We Needed?" (arXiv 2410.01201v1).
    Uses the log-space trick (Section 3.2) for numerical stability:
    b is split into positive/negative parts so log() is well-defined.

    Args:
        log_a: log(1 - z_t), shape [batch, seq_len, dim], always <= 0
        b: z_t * h_tilde_t, shape [batch, seq_len, dim]
        h0: Optional initial hidden state [batch, dim]

    Returns:
        Hidden states [batch, seq_len, dim]
    """
    eps = 1e-38

    if h0 is not None:
        # Prepend h0 as the first element (with log_a=0 so a=1, preserving h0)
        b = torch.cat([h0.unsqueeze(1), b], dim=1)
        log_a = torch.cat([torch.zeros_like(h0.unsqueeze(1)), log_a], dim=1)

    # a_star_t = sum_{i=1}^{t} log(a_i) — cumulative product in log-space
    a_star = torch.cumsum(log_a, dim=1)

    # Split b into positive/negative parts for log-space computation
    b_pos = torch.clamp(b, min=0)
    b_neg = torch.clamp(-b, min=0)

    log_b_pos = torch.log(b_pos + eps)
    h_pos = torch.exp(torch.logcumsumexp(log_b_pos - a_star, dim=1) + a_star)

    log_b_neg = torch.log(b_neg + eps)
    h_neg = torch.exp(torch.logcumsumexp(log_b_neg - a_star, dim=1) + a_star)

    h = h_pos - h_neg

    if h0 is not None:
        h = h[:, 1:]  # Remove the prepended h0 position

    return h


class MinGRUCell(nn.Module):
    """Minimal GRU cell from 'Were RNNs All We Needed?' (arXiv 2410.01201v1).

    Simplifications over standard GRU (Section 3.1):
    - Gate z_t depends only on x_t, not h_{t-1}
    - Candidate h_tilde has no activation function
    - No reset gate

    Recurrence: h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self._linear_z = nn.Linear(input_dim, hidden_dim)
        self._linear_h = nn.Linear(input_dim, hidden_dim)

    @override
    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        """Single-step recurrence: h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t.

        Args:
            x: Input [batch, input_dim]
            h: Previous hidden state [batch, hidden_dim]

        Returns:
            New hidden state [batch, hidden_dim]
        """
        z = torch.sigmoid(self._linear_z(x))
        h_tilde = self._linear_h(x)
        return (1 - z) * h + z * h_tilde

    def parallel(self, x: Tensor, h0: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Parallel scan over a sequence (Section 3.2).

        Computes the same recurrence as forward() but for an entire sequence
        in parallel using log-space cumulative sums.

        Args:
            x: Input sequence [batch, seq_len, input_dim]
            h0: Optional initial hidden state [batch, hidden_dim]

        Returns:
            Tuple of (hidden sequence [batch, seq_len, hidden_dim],
                       final hidden state [batch, hidden_dim])
        """
        k = self._linear_z(x)
        h_tilde = self._linear_h(x)

        # Compute log(z) and log(1-z) in numerically stable form via softplus
        log_z = -F.softplus(-k)  # log(sigmoid(k))
        log_one_minus_z = -F.softplus(k)  # log(1 - sigmoid(k))
        z = torch.exp(log_z)

        h_seq = _parallel_scan_log(log_one_minus_z, z * h_tilde, h0)

        return h_seq, h_seq[:, -1]


class MinGRU(nn.Module):
    """Multi-layer MinGRU for sequence prediction in feature space."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self._num_layers = num_layers

        self._cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self._cells.append(MinGRUCell(in_dim, hidden_dim))

        self._output_proj = nn.Linear(hidden_dim, output_dim)

    @override
    def forward(
        self,
        x: Tensor,
        hidden: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Forward pass over sequence using parallel scan.

        Args:
            x: Input sequence [batch, seq_len, input_dim]
            hidden: List of hidden states per layer, each [batch, hidden_dim].
                   If None, no initial hidden state is used.

        Returns:
            Tuple of:
                - Output sequence [batch, seq_len, output_dim]
                - List of final hidden states per layer
        """
        if hidden is None:
            hidden_inputs: list[Tensor | None] = [None] * self._num_layers
        else:
            hidden_inputs = list(hidden)

        final_hidden: list[Tensor] = []
        layer_input = x
        for i, cell in enumerate(self._cells):
            layer_output, h_final = cast(MinGRUCell, cell).parallel(
                layer_input, hidden_inputs[i]
            )
            final_hidden.append(h_final)
            layer_input = layer_output

        output = self._output_proj(layer_input)

        return output, final_hidden
