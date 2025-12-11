import torch
from torch import nn

from src.models.expansion import RadialBesselBasis, SineBasis
from src.models.layers import MLP


class GeoEmbedding(nn.Module):
    """Geometric feature embedding module for bond and angle features.

    Args:
        num_radial (int): Number of radial basis functions for bond features.
        num_angular (int): Number of angular basis functions for angle features.
        hidden_channels (int): Hidden channel size for the embedding MLPs.
        out_channels (int): Output channel size for the embedded features.
    """

    def __init__(
        self,
        num_radial: int,
        num_angular: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        act: str = "silu",
    ) -> None:
        super().__init__()

        self.rbf = RadialBesselBasis(num_radial=num_radial)
        self.sbf = SineBasis(num_basis=num_angular)

        self.node_embedding = MLP(
            in_channels=num_radial,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            act=act,
        )
        self.edge_embedding = MLP(
            in_channels=num_radial,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            act=act,
        )

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
