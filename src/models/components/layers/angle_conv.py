import torch
from torch import nn


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
        self.attention = nn.Sequential(  # TODO: Change to GATv2
            nn.Linear(angle_input_dim, 1),
            nn.LeakyReLU(negative_slope=0.01), # TODO: change to PRelu
        )
        self.normalized_activation = nn.Sequential(  # TODO: Change to GraphNorm
            nn.LayerNorm(self.angle_fea_len),
            nn.Softplus(),
        )

    def forward(
        self, angle_fea: torch.Tensor, edge_fea: torch.Tensor, nbr_idx: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the ConvAngle module.

        Args:
            angle_fea (torch.Tensor): The angle features.
            edge_fea (torch.Tensor): The edge features.
            nbr_idx (torch.Tensor): The neighbor indices.

        Returns:
            torch.Tensor: The output of the ConvAngle module.
        """
        n, m, o, p = angle_fea.shape

        eij = edge_fea.unsqueeze(2).expand(n, m, m, p)
        eik = edge_fea[nbr_idx, :]
        eijk = torch.cat([eij, eik], dim=3)

        cat_fea = torch.cat([eijk, angle_fea], dim=3)

        output = self.normalized_activation(
            angle_fea + self.attention(cat_fea) * self.linear(cat_fea)
        )

        return output
