import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.models.expansion import GaussianBasis
from src.models.layers import AngleConvLayer, EdgeConvLayer


# TODO: Investigate the influence of the number of pre/post-process layers
# TODO: Investigate the influence of BatchNorm, LayerNorm and GraphNorm in the MP layers
#           https://doi.org/10.48550/arXiv.2009.03294
# TODO: Make use of mini-batch
class CEGANN(nn.Module):
    """Crystal Edge Graph Attention Neural Network (CEGANN) model.

    Implementation based on the paper:
        CEGANN: Crystal Edge Graph Attention Neural Network for multiscale classification of
        materials environment, npj Computational Materials (2023) 9:23.

    Args:
        gbf_bond: Information about the Gaussian basis function expansion for bond features.
        gbf_angle: Information about the Gaussian basis function expansion for angle features.
        n_conv_edge: Number of convolutional layers for edge features.
        edge_expansion_units: Number of units for expanding edge features.
        angle_expansion_units: Number of units for expanding angle features.
        n_classes: Number of output classes.
        embedding: Whether to return embedded features.

    Methods:
        _message_passing(edge_fea, angle_fea, nbr_idx):
            Performs message passing on the edge features and angle features.
        forward(data):
            Forward pass of the CEGANN model.
    """

    def __init__(
        self,
        n_conv_edge: int = 3,
        # n_conv_angle: int | None = None,
        rbf: dict | nn.Module | None = None,
        sbf: dict | nn.Module | None = None,
        edge_expansion_units: int = 128,
        angle_expansion_units: int = 128,
        n_classes: int = 2,
        embedding: bool = False,
    ) -> None:
        super().__init__()

        if rbf is None:
            self.rbf = GaussianBasis()
        elif isinstance(rbf, dict):
            self.rbf = GaussianBasis(**rbf)
        else:
            self.rbf = rbf
        num_edge_features = self.rbf.num_radial

        if sbf is None:
            self.sbf = GaussianBasis()
        elif isinstance(sbf, dict):
            self.sbf = GaussianBasis(**sbf)
        else:
            self.sbf = sbf
        num_angle_features = self.sbf.num_radial

        self.linear_angle = nn.Linear(num_angle_features, angle_expansion_units)
        self.conv_edge = nn.ModuleList(
            [EdgeConvLayer(num_edge_features, num_angle_features) for _ in range(n_conv_edge)]
        )

        n_conv_angle = n_conv_edge - 1
        self.linear_edge = nn.Linear(num_edge_features, edge_expansion_units)
        self.conv_angle = nn.ModuleList(
            [AngleConvLayer(num_edge_features, num_angle_features) for _ in range(n_conv_angle)]
        )

        self.layer_norm = nn.LayerNorm(edge_expansion_units + angle_expansion_units)
        self.softplus = nn.SiLU()
        self.dropout = nn.Dropout()

        self.output_layer = nn.Linear(edge_expansion_units + angle_expansion_units, n_classes)

        self.embedding = embedding

    def _message_passing(
        self,
        edge_features: torch.Tensor,
        angle_features: torch.Tensor,
        neigh_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs hierarchical message passing on the edge and angle features. The edge layer is
        updated before the angle layer. This follows the hierarchy that the bond angles are
        constructed from a pair of edges and any change at the edge level should get propagated to
        the angle level.

        Args:
            # node_features (torch.Tensor): The node features. # DB - GATV2
            edge_features (torch.Tensor): The edge features.
            angle_features (torch.Tensor): The angle features.
            neigh_idx (torch.Tensor): The neighbor indices.

        Returns:
            torch.Tensor: The updated edge features.
            torch.Tensor: The updated angle features.
        """
        edge_features = self.conv_edge[0](edge_features, angle_features, neigh_idx)
        for conv_edge, conv_angle in zip(self.conv_edge[1:], self.conv_angle):
            angle_features = conv_angle(edge_features, angle_features, neigh_idx)
            edge_features = conv_edge(edge_features, angle_features, neigh_idx)

        return edge_features, angle_features

    def forward(self, data: Data) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the CEGANN model.

        Args:
            data: Tuple of bond features, angle features, neighbor indices, and crystal indices.

        Returns:
            torch.Tensor: Output of the model.
            torch.Tensor: Embedded features (if self.embedding is set to True).
        """
        edge_features = data.edge_dist
        angle_features = data.angle_cos
        neigh_idx = data.edge_index
        assert neigh_idx is not None

        # Create features using Gaussian basis function expansion
        edge_features = self.rbf(edge_features)
        angle_features = self.sbf(angle_features, bond=False)  # DB : should be fixed ?

        # Perform message passing
        edge_features, angle_features = self._message_passing(
            edge_features, angle_features, neigh_idx
        )

        # Expand edge features and angle features
        edge_features = self.linear_edge(self.dropout(edge_features))
        angle_features = self.linear_angle(self.dropout(angle_features))

        # Sum over edge features and angle features
        edge_features = torch.sum(self.softplus(edge_features), dim=1)
        angle_features = torch.sum(
            self.softplus(torch.sum(self.softplus(angle_features), dim=2)),
            dim=1,
        )

        # Concatenate edge features and angle features
        crystal_features = torch.cat([edge_features, angle_features], dim=1)

        # Normalize and apply softplus activation
        crystal_features = self.softplus(self.layer_norm(crystal_features))

        if self.embedding:
            embedded_features = crystal_features

        # Apply dropout and linear layer
        output = self.output_layer(self.dropout(crystal_features))

        if self.embedding:
            return output, embedded_features
        else:
            return output
