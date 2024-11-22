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
        n, m = nbr_idx.shape
        # DB : DEBUG
        print("---- New forward ----")
        print(n,m)
        print("input len", edge_fea.size())
        print("squeezed:", edge_fea.unsqueeze(2).size())
        print("method len", self.edge_fea_len)
        # DB : END OF DEBUG

        eij = edge_fea.unsqueeze(2).expand(n, m, m, self.edge_fea_len) # Fails
        eij = edge_fea
        eik = edge_fea[nbr_idx, :]
        print(eij.size(), eik.size(), angle_fea.size())

        cat_fea = torch.cat([eij, eik, angle_fea], dim=3) # Also seems to fail.

        output = self.normalized_activation(
            edge_fea
            + torch.sum(
                self.normalized_activation(self.attention(cat_fea) * self.linear(cat_fea)),
                dim=2,
            )
        )

        return output
