import torch
from torch import nn


class AngleConvLayer(nn.Module):
    """Angle Convolution Layer.

    Args:
        bond_fea_len (int): The length of the bond features.
        angle_fea_len (int): The length of the angle features.
    """

    def __init__(
        self,
        bond_fea_len: int,
        angle_fea_len: int,
    ) -> None:
        super().__init__()

        self.angle_fea_len = angle_fea_len
        self.bond_fea_len = bond_fea_len

        angle_input_dim = self.angle_fea_len + 2 * self.bond_fea_len

        self.linear = nn.Linear(angle_input_dim, self.angle_fea_len)

        self.attention = nn.Sequential(
            nn.Linear(angle_input_dim, 1),
            nn.LeakyReLU(negative_slope=0.01),
        )

        self.normalized_activation = nn.Sequential(
            nn.LayerNorm(self.angle_fea_len),
            nn.Softplus(),
        )

    def forward(
        self,
        bond_features: torch.Tensor,
        angle_features: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the AngleConv module.

        Args:
            bond_features (torch.Tensor): bond features of shape `(n_at * k, n_radial_bond)`.
            angle_features (torch.Tensor): angle features of shape
                `(n_at * k, k - 1, n_radial_angle)`.
            neighbor_indices (torch.Tensor): neighbor indices, shape `(n_at * k, k - 1)`.

        Returns:
            torch.Tensor: The output of the module, of shape `(n_at * k, k - 1, n_radial_angle)`.
        """
        m = neighbor_indices.size(0)  # N_at * k
        n = neighbor_indices.size(1)  # k - 1

        # (n_at * k, k - 1, n_radial_bond)
        eij = bond_features.unsqueeze(1).expand(m, n, self.bond_fea_len)
        eik = bond_features[neighbor_indices]

        # (n_at * k, k - 1, 2 * n_radial_bond)
        eijk = torch.cat([eij, eik], dim=2)

        # (n_at * k, k - 1, 2 * n_radial_bond + n_radial_angle)
        cat_fea = torch.cat([eijk, angle_features], dim=2)

        # output: (n_at * k, k - 1, n_radial_angle)
        output = self.normalized_activation(
            angle_features + self.attention(cat_fea) * self.linear(cat_fea)
        )

        return output
