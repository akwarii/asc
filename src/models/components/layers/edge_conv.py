import torch
from torch import nn


class EdgeConvLayer(nn.Module):
    def __init__(self, edge_fea_len: int, angle_fea_len: int) -> None:
        super().__init__()

        self.edge_fea_len = edge_fea_len
        self.angle_fea_len = angle_fea_len

        edge_input_dim = 2 * self.edge_fea_len + self.angle_fea_len

        self.linear = nn.Linear(edge_input_dim, self.edge_fea_len)
        self.attention = nn.Sequential( # TODO: Change to GATv2
            nn.Linear(edge_input_dim, 1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Softmax(dim=2),
        )
        self.normalized_activation = nn.Sequential(
            nn.LayerNorm(self.edge_fea_len),
            nn.Softplus(),
        )

    def forward(self, edge_fea: torch.Tensor, angle_fea: torch.Tensor, nbr_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ConvEdge module.

        Args:
            edge_fea (torch.Tensor): The input edge features.
            angle_fea (torch.Tensor): The input angle features.
            nbr_idx (torch.Tensor): The neighbor indices.

        Returns:
            torch.Tensor: The output of the ConvEdge module.
        """
        n, m = nbr_idx.shape

        eij = edge_fea.unsqueeze(2).expand(n, m, m, self.edge_fea_len)
        eik = edge_fea[nbr_idx, :]

        cat_fea = torch.cat([eij, eik, angle_fea], dim=3)

        output = self.normalized_activation(
            edge_fea + torch.sum(
                self.normalized_activation(
                    self.attention(cat_fea) *
                    self.linear(cat_fea)
                ),
                dim=2
            )
        )

        return output