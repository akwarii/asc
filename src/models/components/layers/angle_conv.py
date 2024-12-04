import torch
from torch import nn
from torch_geometric.nn.norm import GraphNorm # DB
from torch_geometric.nn.conv import GATv2Conv # DB


class AngleConvLayer(nn.Module):
    """Angle Convolution Layer.

    Args:
        edge_fea_len (int): The length of the edge features.
        angle_fea_len (int): The length of the angle features.
    """

    def __init__(
        self,
        edge_fea_len: int,
        angle_fea_len: int,
    ):
        super().__init__()

        self.angle_fea_len = angle_fea_len
        self.edge_fea_len = edge_fea_len

        angle_input_dim = self.angle_fea_len + 2 * self.edge_fea_len

        self.linear = nn.Linear(angle_input_dim, self.angle_fea_len)

        # self.attention = GATv2Conv(angle_input_dim,1) # DB - GATV2
        self.attention = nn.Sequential(  # TODO: Change to GATv2
            nn.Linear(angle_input_dim, 1),
            # # nn.LeakyReLU(negative_slope=0.01),  # TODO: change to PRelu
            nn.PReLU(), # DB
            # nn.PReLU(num_parameters=20), # DB
        )
        
        self.normalized_activation = nn.Sequential(  # TODO: Change to GraphNorm
            # nn.LayerNorm(self.angle_fea_len),
            GraphNorm(self.angle_fea_len), # DB
            nn.Softplus(),
        )

    def forward(
        self, 
        # node_fea: torch.Tensor, # DB - GATV2
        edge_fea: torch.Tensor, 
        angle_fea: torch.Tensor, 
        nbr_idx: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the ConvAngle module.

        Args:
            # node_fea (torch.Tensor): The node features. # DB - GATV2
            edge_fea (torch.Tensor): The edge features.
            angle_fea (torch.Tensor): The angle features.
            nbr_idx (torch.Tensor): The neighbor indices.

        Returns:
            torch.Tensor: The output of the ConvAngle module.
        """
        # DB
        # t, n = torch.unique_consecutive(nbr_idx[0], return_counts=True)
        # n = n[t == 1][0].item() # Number of neighbors, to avoid issues with monoatomic boxes
        # m = nbr_idx.size()[1] // n # Number of atoms
        # _nbr_idx = torch.reshape(
        #     nbr_idx[1],
        #     (m,n)
        # )
        # # Edge features with the correct shape
        # _edge_fea = torch.reshape(
        #     edge_fea,
        #     (m, n, edge_fea.size()[-1])
        # )
        # # Angle features with the correct shape
        # _angle_fea = torch.reshape(
        #     angle_fea,
        #     (m, n, n, angle_fea.size()[-1])
        # )
        # print("ANGLE", node_fea.size())
        # print("ANGLE", _nbr_idx.size())
        # print("ANGLE", _edge_fea.size())
        # print("ANGLE", _angle_fea.size())
        # output = self.normalized_activation(
        #     _angle_fea
        #     + self.attention(
        #         x=_edge_fea,
        #         edge_index=_nbr_idx,
        #         edge_attr=_angle_fea
        #     )
        # )
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
        # n, m, o, p = angle_fea.shape
        # eij = edge_fea.unsqueeze(2).expand(n, m, m, p)
        # eik = edge_fea[nbr_idx, :]
        # eijk = torch.cat([eij, eik], dim=3)
        # cat_fea = torch.cat([eijk, angle_fea], dim=3)
        # output = self.normalized_activation(
        #     angle_fea + self.attention(cat_fea) * self.linear(cat_fea)
        # )

        # Modified
        n, m, _, p = _angle_fea.shape
        eij = _edge_fea.unsqueeze(2).expand(n, m, m, p)
        eik = _edge_fea[_nbr_idx, :]
        eijk = torch.cat([eij, eik], dim=3)
        cat_fea = torch.cat([eijk, _angle_fea], dim=3)
        output = self.normalized_activation(
            _angle_fea + self.attention(cat_fea) * self.linear(cat_fea)
        )

        return output
