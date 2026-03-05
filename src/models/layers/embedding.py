from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

from src.models.expansion import RadialBesselBasis, SineBasis
from src.models.layers import MLP


class GeometricEmbedding(nn.Module):
    """Geometric feature embedding module for bond and angle features.

    Args:
        num_radial (int): Number of radial basis functions for bond features.
        num_angular (int): Number of angular basis functions for angle features.
        num_channels (int): Channel size for the embedded node, edges and hidden features.
        num_layers (int, optional): Number of layers in the embedding MLPs. Default is 1.
        act (str | Callable | None, optional): Activation function for the embedding MLPs.
            Default is "silu".
        act_kwargs (dict | None, optional): Additional arguments for the activation function.
            Default is None.
    """

    def __init__(
        self,
        num_radial: int,
        num_angular: int,
        num_channels: int,
        num_layers: int = 1,
        act: str | Callable | None = "silu",
        act_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.rbf = RadialBesselBasis(num_radial=num_radial)
        self.sbf = SineBasis(num_basis=num_angular)
        self.node_embedding = MLP(
            in_channels=num_radial,
            hidden_channels=num_channels,
            num_layers=num_layers,
            out_channels=num_channels,
            act=act,
            act_kwargs=act_kwargs,
        )
        self.edge_embedding = MLP(
            in_channels=num_angular,
            hidden_channels=num_channels,
            num_layers=num_layers,
            out_channels=num_channels,
            act=act,
            act_kwargs=act_kwargs,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset model parameters."""
        if hasattr(self.rbf, "reset_parameters"):
            self.rbf.reset_parameters()
        if hasattr(self.sbf, "reset_parameters"):
            self.sbf.reset_parameters()
        self.node_embedding.reset_parameters()
        self.edge_embedding.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_attr: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass of the GeoEmbedding module.

        Args:
            x (Tensor): node features of shape `(num_nodes, in_node_channels)`.
            edge_attr (Tensor): edge features of shape `(num_edges, in_edge_channels)`.

        Returns:
            tuple[Tensor, Tensor]: The embedded node and edge features.
        """
        x_rbf = self.rbf(x)
        x_emb = self.node_embedding(x_rbf)

        edge_attr_sbf = self.sbf(edge_attr)
        edge_attr_emb = self.edge_embedding(edge_attr_sbf)

        return x_emb, edge_attr_emb
