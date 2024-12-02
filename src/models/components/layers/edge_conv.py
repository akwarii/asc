import torch
from torch import nn


class EdgeConvLayer(nn.Module):
    def __init__(self, edge_fea_len: int, angle_fea_len: int) -> None:
        """Edge Convolution Layer.

        Args:
            edge_fea_len (int): The length of the edge features.
            angle_fea_len (int): The length of the angle features.
        """
        super().__init__()

        self.edge_fea_len = edge_fea_len
        self.angle_fea_len = angle_fea_len

        edge_input_dim = 2 * self.edge_fea_len + self.angle_fea_len

        self.linear = nn.Linear(edge_input_dim, self.edge_fea_len)
        self.attention = nn.Sequential(  # TODO: Change to GATv2
            nn.Linear(edge_input_dim, 1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Softmax(dim=2),
        )
        self.normalized_activation = nn.Sequential(  # TODO: Change to GraphNorm
            nn.LayerNorm(self.edge_fea_len),
            nn.Softplus(),
        )

    def forward(
        self, edge_fea: torch.Tensor, angle_fea: torch.Tensor, nbr_idx: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the ConvEdge module.

        Args:
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
        # "Data" object to build its graphs. In such cases the aformentioned
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
        n = n[t == 1][0].item() # Number of neighbors, to avoid issues with monoatomic boxes
        m = nbr_idx.size()[1] // n # Number of atoms
        _nbr_idx = torch.reshape(
            nbr_idx[1],
            (m,n)
        )
        # Edge features with the correct shape
        _edge_fea = torch.reshape(
            edge_fea,
            (m, n, edge_fea.size()[-1])
        )
        # Angle features with the correct shape
        _angle_fea = torch.reshape(
            angle_fea,
            (m, n, n, angle_fea.size()[-1])
        )        

        # Original code
        # n, m = nbr_idx.shape
        # eij = edge_fea.unsqueeze(2).expand(n, m, m, self.edge_fea_len)
        # eik = edge_fea[nbr_idx, :]
        # cat_fea = torch.cat([eij, eik, angle_fea], dim=3)

        # Modified code # DB
        n,m = _nbr_idx.shape
        eij = _edge_fea.unsqueeze(2).expand(n, m, m, self.edge_fea_len)
        eik = _edge_fea[_nbr_idx, :]
        # print(eij.size(), eik.size(), _angle_fea.size()) # DEBUG
        cat_fea = torch.cat([eij, eik, _angle_fea], dim=3)

        output = self.normalized_activation(
            # edge_fea
            _edge_fea # DB
            + torch.sum(
                self.normalized_activation(self.attention(cat_fea) * self.linear(cat_fea)),
                dim=2,
            )
        )

        return output
