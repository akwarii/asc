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
        self,
        bond_x: Tensor,
        num_atoms: int | Tensor,
        bond_source: Tensor | None = None,
        bond_target: Tensor | None = None,
    ) -> Tensor:
        """Forward pass of the readout layer.
        
        Args:
            bond_x: Bond embeddings tensor.
            num_atoms: Total number of atoms (can be int or 0-d Tensor to avoid graph breaks).
            bond_source: Source atom indices for each bond.
            bond_target: Target atom indices for each bond.
        """
        if self.incidence == "out":
            assert bond_source is not None, "bond_source is required for 'out' incidence."
            return scatter(bond_x, bond_source, dim=0, dim_size=num_atoms, reduce=self.reduce)

        if self.incidence == "in":
            assert bond_target is not None, "bond_target is required for 'in' incidence."
            return scatter(bond_x, bond_target, dim=0, dim_size=num_atoms, reduce=self.reduce)

        if self.incidence == "both":
            assert (
                bond_source is not None and bond_target is not None
            ), "bond_source and bond_target are required for 'both' incidence."
            bond_idx = torch.cat([bond_source, bond_target], dim=0)
            bond_x = torch.cat([bond_x, bond_x], dim=0)
            return scatter(bond_x, bond_idx, dim=0, dim_size=num_atoms, reduce=self.reduce)

        raise ValueError(f"Invalid incidence type: {self.incidence}")
