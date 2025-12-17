from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.models.expansion import RadialBesselBasis, SineBasis
from src.models.layers import MLP


@dataclass
class EmbeddingConfig:
    """Configuration for the GeometricEmbedding module.

    Args:
        num_radial (int): Number of radial basis functions for bond features.
        num_angular (int): Number of angular basis functions for angle features.
        node_out_channels (int): Output channel size for the embedded node features.
        edge_out_channels (int): Output channel size for the embedded edge features.
        num_layers (int, optional): Number of layers in the embedding MLPs. Default is 1.
        hidden_channels (int | None, optional): Hidden channel size for the embedding MLPs.
            Default is None.
        act (str | Callable | None, optional): Activation function for the embedding MLPs.
            Default is "silu".
        act_kwargs (dict | None, optional): Additional arguments for the activation function.
            Default is None.
    """

    num_radial: int
    num_angular: int
    node_out_channels: int
    edge_out_channels: int
    num_layers: int = 1
    hidden_channels: int | None = None
    act: str | Callable | None = "silu"
    act_kwargs: dict[str, Any] | None = None


class GeometricEmbedding(nn.Module):
    """Geometric feature embedding module for bond and angle features.

    Args:
        config (EmbeddingConfig): Configuration for the embedding module.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__()

        self.rbf = RadialBesselBasis(num_radial=config.num_radial)
        self.sbf = SineBasis(num_basis=config.num_angular)
        self.node_embedding = MLP(
            in_channels=config.num_radial,
            hidden_channels=config.hidden_channels,
            num_layers=config.num_layers,
            out_channels=config.node_out_channels,
            act=config.act,
            act_kwargs=config.act_kwargs,
        )
        self.edge_embedding = MLP(
            in_channels=config.num_angular,
            hidden_channels=config.hidden_channels,
            num_layers=config.num_layers,
            out_channels=config.edge_out_channels,
            act=config.act,
            act_kwargs=config.act_kwargs,
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
        x: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the GeoEmbedding module.

        Args:
            x (torch.Tensor): node features of shape `(num_nodes, in_node_channels)`.
            edge_attr (torch.Tensor): edge features of shape `(num_edges, in_edge_channels)`.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The embedded node and edge features.
        """
        x_rbf = self.rbf(x)
        x_emb = self.node_embedding(x_rbf)

        edge_attr_sbf = self.sbf(edge_attr)
        edge_attr_emb = self.edge_embedding(edge_attr_sbf)

        return x_emb, edge_attr_emb
