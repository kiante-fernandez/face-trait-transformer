"""MLP regression head. Same shape shipped in every bundle."""
from __future__ import annotations

import torch
from torch import nn


class TraitHead(nn.Module):
    """MLP head that maps a backbone embedding to a 34-d trait vector.

    Parameters
    ----------
    in_dim   : feature dim of the backbone CLS token (768/1024/1536 for DINOv2 B/L/G).
    out_dim  : number of output traits (default 34, matching OMI).
    hidden   : MLP hidden width. If None, this is a linear probe.
    dropout  : dropout after the GELU nonlinearity.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 34,
        hidden: int | None = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden is None:
            self.net: nn.Module = nn.Linear(in_dim, out_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
