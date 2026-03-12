from typing import override

import torch
import torch.nn as nn
from torch import Tensor


class MinGRUCell(nn.Module):
    """Minimal GRU cell without reset gate."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self._linear_z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self._linear_h = nn.Linear(input_dim + hidden_dim, hidden_dim)

    @override
    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input [batch, input_dim]
            h: Hidden state [batch, hidden_dim]

        Returns:
            New hidden state [batch, hidden_dim]
        """
        combined = torch.cat([x, h], dim=-1)
        z = torch.sigmoid(self._linear_z(combined))
        h_candidate = torch.tanh(self._linear_h(combined))
        return (1 - z) * h + z * h_candidate


class MinGRU(nn.Module):
    """Minimal GRU for sequence prediction in feature space."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
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
        """Forward pass over sequence.

        Args:
            x: Input sequence [batch, seq_len, input_dim]
            hidden: List of hidden states per layer, each [batch, hidden_dim].
                   If None, initialized to zeros.

        Returns:
            Tuple of:
                - Output sequence [batch, seq_len, output_dim]
                - List of final hidden states per layer
        """
        batch_size, seq_len, _ = x.shape

        if hidden is None:
            hidden = [
                torch.zeros(batch_size, self._hidden_dim, device=x.device)
                for _ in range(self._num_layers)
            ]

        outputs = []
        for t in range(seq_len):
            inp = x[:, t]
            new_hidden = []
            for i, cell in enumerate(self._cells):
                h = cell(inp, hidden[i])
                new_hidden.append(h)
                inp = h
            hidden = new_hidden
            outputs.append(inp)

        output = torch.stack(outputs, dim=1)  # [batch, seq_len, hidden_dim]
        output = self._output_proj(output)  # [batch, seq_len, output_dim]

        return output, hidden
