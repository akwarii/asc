import torch
import torch.nn as nn

from src.models.components.expansion.radial import GaussianBasis
from src.models.components.layers.angle_conv import AngleConvLayer
from src.models.components.layers.edge_conv import EdgeConvLayer


# TODO: Investigate the influence of the number of pre/post-process layers
# TODO: Investigate the influence of BatchNorm, LayerNorm and GraphNorm in the MP layers
#           https://doi.org/10.48550/arXiv.2009.03294
# TODOX: Make use of mini-batch
class CEGANN(nn.Module):
    """
    Crystal Edge Graph Attention Neural Network (CEGANN) model.
    Implementation based on the paper: https://doi.org/10.1038/s41524-023-00975-z

    Args:
        gbf_bond (dict): Dictionary containing information about the Gaussian basis function expansion for bond features.
        gbf_angle (dict): Dictionary containing information about the Gaussian basis function expansion for angle features.
        n_conv_edge (int): Number of convolutional layers for edge features.
        edge_expansion_units (int): Number of units for expanding edge features.
        angle_expansion_units (int): Number of units for expanding angle features.
        n_classes (int): Number of output classes.
        pooling (bool): Whether to perform pooling on crystal features.
        embedding (bool): Whether to return embedded features.

    Methods:
        _message_passing(edge_fea, angle_fea, nbr_idx):
            Performs message passing on the edge features and angle features.
        forward(data):
            Forward pass of the CEGANN model.
        pool(atom_fea, crys_idx):
            Pooling function for crystal features.
    """

    def __init__(
        self,
        gbf_bond: dict,
        gbf_angle: dict,
        n_conv_edge: int = 3,
        n_conv_angle: int = None, # DB
        edge_expansion_units: int = 128,
        angle_expansion_units: int = 128,
        n_classes: int = 2,
        pooling: bool = False,
        embedding: bool = False,
    ) -> None:
        super().__init__()

        self.pooling = pooling
        self.embedding = embedding

        edge_features_len = gbf_bond["num_radial"]
        angle_features_len = gbf_angle["num_radial"]

        # edge_features_len = gbf_bond["steps"] # DB: comment ?
        # angle_features_len = gbf_angle["steps"] # DB: comment ?
        # edge_features_len = gbf_bond.pop("steps") # DB as Gaussian Basis does not accept steps
        # angle_features_len = gbf_angle.pop("steps") # DB as Gaussian Basis does not accept steps

        self.gbf_edge = GaussianBasis(**gbf_bond) # ** added by DB
        self.linear_angle = nn.Linear(angle_features_len, angle_expansion_units)
        self.conv_edge = nn.ModuleList(
            [EdgeConvLayer(edge_features_len, angle_features_len) for _ in range(n_conv_edge)]
        )

        self.gbf_angle = GaussianBasis(**gbf_angle) # ** added by DB
        self.linear_edge = nn.Linear(edge_features_len, edge_expansion_units)
        if n_conv_angle is None : n_conv_angle = n_conv_edge - 1 # DB
        self.conv_angle = nn.ModuleList(
            [AngleConvLayer(edge_features_len, angle_features_len) for _ in range(n_conv_angle)]
        )

        self.layer_norm = nn.LayerNorm(
            edge_expansion_units + angle_expansion_units
        )  # TODO: change to GraphNorm
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout()

        self.output_layer = nn.Linear(edge_expansion_units + angle_expansion_units, n_classes)

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
            edge_features (torch.Tensor): The edge features.
            angle_features (torch.Tensor): The angle features.
            neigh_idx (torch.Tensor): The neighbor indices.

        Returns:
            torch.Tensor: The updated edge features.
            torch.Tensor: The updated angle features.
        """
        edge_features = self.conv_edge[0](edge_features, angle_features, neigh_idx)
        for conv_edge, conv_angle in zip(self.conv_edge[1:], self.conv_angle):
            angle_features = conv_angle(angle_features, edge_features, neigh_idx)
            edge_features = conv_edge(edge_features, angle_features, neigh_idx)

        return edge_features, angle_features

    def forward(self, data: tuple) -> torch.Tensor | tuple[torch.Tensor]:
        """Forward pass of the CEGANN model.

        Args:
            data (tuple): Tuple containing bond features, angle features, neighbor indices, and crystal indices.

        Returns:
            torch.Tensor: Output of the model.
            torch.Tensor: Embedded features (if self.embedding is set to True).
        """
        # returns Data(edge_index=[2, 37932], pos=[2022, 3], num_nodes=2022, cell=[192, 3], edge_dist=[37932], angle_cos=[37932, 19])
        neigh_idx, _, _, _, edge_features, angle_features = [ d[1] for d in data ] # DB, `data` returns tuples
        # edge_features, angle_features, neigh_idx, crystal_idx = data

        # Create features using Gaussian basis function expansion
        edge_features = self.gbf_edge(edge_features)
        angle_features = self.gbf_angle(angle_features, bond=False) # DB : should be fixed ?

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

        if self.pooling:
            crystal_features = self.pool(crystal_features, crystal_idx)

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

    def pool(self, atom_fea, crys_idx):
        """Pooling function for crystal features.

        Args:
            atom_fea (torch.Tensor): Atom-level features.
            crys_idx (list): List of indices mapping crystal features to atom features.

        Returns:
            torch.Tensor: Pooled crystal features.
        """
        summed_fea = [
            torch.mean(atom_fea[idx_map[0] : idx_map[1], :], dim=0, keepdim=True)
            for idx_map in crys_idx
        ]
        return torch.cat(summed_fea, dim=0)
