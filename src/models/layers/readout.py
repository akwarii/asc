from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import scatter


class BondToAtomReadout(nn.Module):
    """Aggregate bond (original-edge) embeddings -> atom (original-node) embeddings.

    Args:
        reduce: Reduction method to use (`sum`, `mean`, `max`).
        incidence: Type of incidence to consider (`in`, `out`, `both`).
            `in`: aggregate messages from incoming bonds,
            `out`: aggregate messages from outgoing bonds,
            `both`: aggregate messages from both incoming and outgoing bonds.
    """

    def __init__(
        self,
        reduce: Literal["sum", "mean", "max"] = "mean",
        incidence: Literal["in", "out", "both"] = "out",
    ) -> None:
        super().__init__()
        self.reduce = reduce
        self.incidence = incidence

    def forward(
        self, bond_x: Tensor, bond_source: Tensor, bond_target: Tensor, num_atoms: int
    ) -> Tensor:
        """Forward pass of the readout layer."""
        if self.incidence == "out":
            return scatter(bond_x, bond_source, dim=0, dim_size=num_atoms, reduce=self.reduce)

        if self.incidence == "in":
            return scatter(bond_x, bond_target, dim=0, dim_size=num_atoms, reduce=self.reduce)

        if self.incidence == "both":
            bond_idx = torch.cat([bond_source, bond_target], dim=0)
            bond_x = torch.cat([bond_x, bond_x], dim=0)
            return scatter(bond_x, bond_idx, dim=0, dim_size=num_atoms, reduce=self.reduce)

        raise ValueError(f"Invalid incidence type: {self.incidence}")
