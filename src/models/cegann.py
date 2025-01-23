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
        n_conv_edge: Number of convolutional layers for edge features.
        rbf: Information about the Gaussian basis function expansion for bond features.
        sbf: Information about the Gaussian basis function expansion for angle features.
        edge_expansion_units: Number of units for expanding edge features.
        angle_expansion_units: Number of units for expanding angle features.
        n_classes: Number of output classes.
        embedding: Whether to return embedded features.
    """

    def __init__(
        self,
        n_classes: int,
        n_bond_conv: int = 3,
        rbf: dict | nn.Module | None = None,
        sbf: dict | nn.Module | None = None,
        edge_expansion_units: int = 128,
        angle_expansion_units: int = 128,
        dropout: float = 0.1,
        embedding: bool = False,
    ) -> None:
        super().__init__()

        if rbf is None:
            self.rbf = GaussianBasis()
        elif isinstance(rbf, dict):
            rbf.pop("bond", None)
            self.rbf = GaussianBasis(**rbf)
        else:
            self.rbf = rbf
        n_bond_features = self.rbf.num_radial

        if sbf is None:
            self.sbf = GaussianBasis(bond=False)
        elif isinstance(sbf, dict):
            sbf.pop("bond", None)
            self.sbf = GaussianBasis(bond=False, **sbf)
        else:
            self.sbf = sbf
        n_angle_features = self.sbf.num_radial

        assert self.rbf.bond is True
        assert self.sbf.bond is False

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
            channel_list=[cat_units]
            # + [hidden_channels]
            + [n_classes],
            dropout=dropout,
            act=nn.SiLU,
        )

        self.embedding = embedding

    def _message_passing(
        self,
        bond_features: torch.Tensor,
        angle_features: torch.Tensor,
        neigh_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs hierarchical message passing on the edge and angle features. The edge layer is
        updated before the angle layer. This follows the hierarchy that the bond angles are
        constructed from a pair of edges and any change at the edge level should get propagated to
        the angle level.

        Args:
            bond_features (torch.Tensor): The bond features.
            angle_features (torch.Tensor): The angle features.
            neigh_idx (torch.Tensor): The neighbor indices.

        Returns:
            torch.Tensor: The updated bond features of shape `(n_at * k, n_radial_bond)`.
            torch.Tensor: The updated angle features of shape `(n_at * k, k - 1, n_radial_angle)`.
        """
        n = angle_features.size(1)  # k-1
        m = neigh_idx.size(1) // n  # N_at * k
        neigh_idx = torch.reshape(neigh_idx[1], (m, n))

        bond_features = self.bond_conv[0](bond_features, angle_features, neigh_idx)
        for conv_edge, conv_angle in zip(self.bond_conv[1:], self.angle_conv):
            angle_features = conv_angle(bond_features, angle_features, neigh_idx)
            bond_features = conv_edge(bond_features, angle_features, neigh_idx)

        return bond_features, angle_features

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

        bond_features, neigh_idx, angle_features = data.x, data.edge_index, data.edge_attr

        # Create features using Gaussian basis function expansion
        bond_features = self.rbf(bond_features)
        angle_features = self.sbf(angle_features)

        # Perform message passing
        bond_features, angle_features = self._message_passing(
            bond_features, angle_features, neigh_idx
        )

        # Expand edge features and angle features
        bond_features = self.linear_bond(bond_features)
        angle_features = self.linear_angle(angle_features)

        # Reshape bond features and angle features
        # This is useful as we want to sum over the k neighbors later
        # while PyG LineGraph implied stacking the features of the k neighbors.
        k = angle_features.size(1) + 1
        bond_features = bond_features.reshape(
            bond_features.size(0) // k, k, bond_features.size(-1)
        )
        angle_features = angle_features.reshape(
            angle_features.size(0) // k, k, k - 1, angle_features.size(-1)
        )

        # Sum over edge features and angle features
        bond_features = torch.sum(self.softplus(bond_features), dim=1)
        angle_features = torch.sum(
            self.softplus(torch.sum(self.softplus(angle_features), dim=2)),
            dim=1,
        )

        # Concatenate edge features and angle features
        crystal_features = torch.cat([bond_features, angle_features], dim=1)

        # Normalize and apply softplus activation
        crystal_features = self.softplus(self.layer_norm(crystal_features))

        if self.embedding:
            embedded_features = crystal_features

        # Apply dropout and linear layer
        output = self.classification_head(crystal_features)

        if self.embedding:
            return output, embedded_features
        else:
            return output
