import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MLP

from src.models.expansion import GaussianBasis
from src.models.layers import AngleConvLayer, BondConvLayer


class CEGANN(nn.Module):
    """Crystal Edge Graph Attention Neural Network (CEGANN) model.

    Implementation based on the paper:
        CEGANN: Crystal Edge Graph Attention Neural Network for multiscale classification of
        materials environment, npj Computational Materials (2023) 9:23.

    Args:
        out_channels: Number of output classes.
        n_conv_edge: Number of convolutional layers for edge features.
        rbf: Information about the Gaussian basis function expansion for bond features.
        sbf: Information about the Gaussian basis function expansion for angle features.
        edge_expansion_units: Number of units for expanding edge features.
        angle_expansion_units: Number of units for expanding angle features.
    """

    def __init__(
        self,
        out_channels: int,
        *,
        n_bond_conv: int = 3,
        rbf: dict | nn.Module | None = None,
        sbf: dict | nn.Module | None = None,
        num_radial: int | None = None,
        edge_expansion_units: int = 128,
        angle_expansion_units: int = 128,
        dropout: float = 0.1,
        classification_units: int = 128,
        classification_layers: int = 2,
    ) -> None:
        super().__init__()

        if (rbf is None or sbf is None) and num_radial is None:
            raise ValueError("num_radial must be provided if rbf and/or sbf are not provided.")

        if rbf is None:
            assert num_radial is not None
            self.rbf = GaussianBasis(num_radial=num_radial)
        elif isinstance(rbf, dict):
            rbf.pop("bond", None)
            self.rbf = GaussianBasis(**rbf)
        else:
            self.rbf = rbf
        n_bond_features: int = self.rbf.num_radial  # type: ignore

        if sbf is None:
            assert num_radial is not None
            self.sbf = GaussianBasis(num_radial=num_radial)
        elif isinstance(sbf, dict):
            sbf.pop("bond", None)
            self.sbf = GaussianBasis(**sbf)
        else:
            self.sbf = sbf
        n_angle_features: int = self.sbf.num_radial  # type: ignore

        self.linear_bond = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(n_bond_features, edge_expansion_units)
        )
        self.bond_conv = nn.ModuleList(
            [BondConvLayer(n_bond_features, n_angle_features) for _ in range(n_bond_conv)]
        )

        n_angle_conv = n_bond_conv - 1
        self.linear_angle = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(n_angle_features, angle_expansion_units)
        )
        self.angle_conv = nn.ModuleList(
            [AngleConvLayer(n_bond_features, n_angle_features) for _ in range(n_angle_conv)]
        )

        cat_units = edge_expansion_units + angle_expansion_units
        self.layer_norm = nn.LayerNorm(cat_units)
        self.softplus = nn.Softplus()

        self.classification_head = MLP(
            in_channels=cat_units,
            hidden_channels=classification_units,
            num_layers=classification_layers,
            out_channels=out_channels,
            dropout=dropout,
            plain_last=True,
        )

    def _message_passing(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs hierarchical message passing on the edge and angle features. The edge layer is
        updated before the angle layer. This follows the hierarchy that the bond angles are
        constructed from a pair of edges and any change at the edge level should get propagated to
        the angle level.

        Args:
            x (torch.Tensor): The bond features ie the node attributes of the line graph.
            edge_attr (torch.Tensor): The angle features ie the edge attributes of the line graph.
            edge_index (torch.Tensor): The neighbor indices.

        Returns:
            torch.Tensor: The updated bond features of shape `(n_at * k, n_radial_bond)`.
            torch.Tensor: The updated angle features of shape `(n_at * k, k - 1, n_radial_angle)`.
        """
        num_nodes, num_edges = x.size(0), edge_attr.size(0)
        k = num_edges // num_nodes + 1
        neigh_index = torch.reshape(edge_index[1], (num_nodes, k - 1))

        x = self.bond_conv[0](x, edge_attr, neigh_index)
        for conv_edge, conv_angle in zip(self.bond_conv[1:], self.angle_conv):
            edge_attr = conv_angle(x, edge_attr, neigh_index)
            x = conv_edge(x, edge_attr, neigh_index)

        return x, edge_attr

    def forward(self, data: Data) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the CEGANN model.

        Args:
            data: Tuple of bond features, angle features, neighbor indices, and crystal indices.

        Returns:
            torch.Tensor: Output of the model.
            torch.Tensor: Embedded features (if self.embedding is set to True).
        """
        assert data.x is not None
        assert data.edge_attr is not None
        assert data.edge_index is not None

        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index

        # Create features using Gaussian basis function expansion
        x = self.rbf(x)
        edge_attr = self.sbf(edge_attr)

        # Perform message passing
        x_m, edge_attr_m = self._message_passing(x, edge_attr, edge_index)

        # Expand edge features and angle features
        x_l = self.linear_bond(x_m)
        edge_attr_l = self.linear_angle(edge_attr_m)

        # Reshape bond features and angle features
        # This is useful as we want to sum over the k neighbors later
        # while PyG LineGraph implied stacking the features of the k neighbors.
        k = edge_attr.size(0) // x.size(0) + 1
        num_atoms = x.size(0) // k

        x = x_l.view(num_atoms, k, x_l.size(-1))
        edge_attr = edge_attr_l.view(num_atoms, k, k - 1, edge_attr_l.size(-1))

        # Aggregate over neighbors
        x = torch.sum(self.softplus(x), dim=1)
        edge_attr = torch.sum(
            self.softplus(torch.sum(self.softplus(edge_attr), dim=2)),
            dim=1,
        )

        # Concatenate edge features and angle features
        node_repr = torch.cat([x, edge_attr], dim=-1)

        # Normalize and apply softplus activation
        #? WHY
        embedding = self.softplus(self.layer_norm(node_repr))

        #TODO add readout for optional graph-level prediction here

        # Apply dropout and linear layer
        output = self.classification_head(embedding)

        return output
