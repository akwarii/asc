import torch
from torch import nn
from torch_geometric.nn.norm import GraphNorm  # DB


class EdgeConvLayer(nn.Module):
    """Edge Convolution Layer.

    Attributes:
        edge_fea_len: The number of edge features.
        angle_fea_len: The number of angular features
        linear: A linear layer.
        attention: An attention layer for the edge features. It is made of a linear layer followed
            by a PReLU and softmax layers.
        normalized_activation: A layer that normalizes the output of the attention layer using a
        GraphNorm and applies a softplus activation.
    """

    def __init__(self, edge_fea_len: int, angle_fea_len: int) -> None:
        """Initializes the EdgeConvLayer module.

        Args:
            edge_fea_len: The number of edge features.
            angle_fea_len: The number of angular features
        """
        super().__init__()

        self.edge_fea_len = edge_fea_len
        self.angle_fea_len = angle_fea_len

        edge_input_dim = 2 * self.edge_fea_len + self.angle_fea_len

        self.linear = nn.Linear(edge_input_dim, self.edge_fea_len)

        # self.attention = GATv2Conv(edge_fea_len, 1) # DB
        self.attention = nn.Sequential(  # TODO: Change to GATv2
            nn.Linear(edge_input_dim, 1),
            # # nn.LeakyReLU(negative_slope=0.01),
            # nn.PReLU(num_parameters=20), # DB
            nn.PReLU(),  # DB
            nn.Softmax(dim=2),
        )

        self.normalized_activation = nn.Sequential(  # TODO: Change to GraphNorm
            # nn.LayerNorm(self.edge_fea_len),
            GraphNorm(self.edge_fea_len),
            nn.Softplus(),  # DB
        )

    def forward(
        self,
        # node_fea: torch.Tensor, # DB- GATV2
        edge_fea: torch.Tensor,
        angle_fea: torch.Tensor,
        nbr_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the ConvEdge module.

        Args:
            # node_fea (torch.Tensor): The input node features. # DB - GATV2
            edge_fea (torch.Tensor): The input edge features.
            angle_fea (torch.Tensor): The input angle features.
            nbr_idx (torch.Tensor): The neighbor indices.

        Returns:
            torch.Tensor: The output of the ConvEdge module.
        """
        ############################### NOTE ###############################
        # In comparison to the original CEGANN framework, the tensors used
        # here have a different build.
        # In the original CEGANN :
        #   * nbr_idx.shape   = [N_atoms, N_neigh]
        #   * edge_fea.shape  = [N_atoms, N_neigh, N_rad_features]
        #   * angle_fea.shape = [N_atoms, N_neigh, N_neigh, N_ang_features]
        # Meanwhile, our current framework uses the torch_geometric.data
        # "Data" object to build its graphs. In such cases the aforementioned
        # tensors must have the following shape instead :
        #   * nbr_idx.shape   = [N_atoms*N_neigh,2]
        #   * edge_fea.shape  = [N_atoms*N_neigh, N_rad_features]
        #   * angle_fea.shape = [N_atoms*N_neigh*N_neigh, N_ang_features]
        # This means we first need to rework those tensors to the shape used
        # in the original framework if we want to use similar layer inputs
        # and outputs.
        # The following lines attempt to do just that in as an efficient way
        # as I can think of. ~DB
        ############################### NOTE ###############################

        # Reshaping the tensors
        t, n = torch.unique_consecutive(nbr_idx[0], return_counts=True)
        n = n[t == 1][0].detach()  # Number of neighbors, to avoid issues with monoatomic boxes
        m = nbr_idx.size()[1] // n  # Number of atoms
        _nbr_idx = torch.reshape(nbr_idx[1], (m, n))

        # Edge features with the correct shape
        _edge_fea = torch.reshape(edge_fea, (m, n, edge_fea.size()[-1]))

        # Angle features with the correct shape
        _angle_fea = torch.reshape(angle_fea, (m, n, n, angle_fea.size()[-1]))

        # Modified code # DB
        n, m = _nbr_idx.shape
        eij = _edge_fea.unsqueeze(2).expand(n, m, m, self.edge_fea_len)
        eik = _edge_fea[_nbr_idx, :]
        cat_fea = torch.cat([eij, eik, _angle_fea], dim=3)

        output = self.normalized_activation(
            # edge_fea
            _edge_fea  # DB
            + torch.sum(
                self.normalized_activation(self.attention(cat_fea) * self.linear(cat_fea)),
                dim=2,
            )
        )

        return output
