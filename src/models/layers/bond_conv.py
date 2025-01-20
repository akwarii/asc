import torch
from torch import nn


class BondConvLayer(nn.Module):
    """Bond Convolution Layer.

    Args:
        bond_fea_len: The number of bond features.
        angle_fea_len: The number of angular features
    """

    def __init__(self, bond_fea_len: int, angle_fea_len: int) -> None:
        super().__init__()

        self.bond_fea_len = bond_fea_len
        self.angle_fea_len = angle_fea_len

        bond_input_dim = 2 * self.bond_fea_len + self.angle_fea_len

        self.linear = nn.Linear(bond_input_dim, self.bond_fea_len)

        self.attention = nn.Sequential(
            nn.Linear(bond_input_dim, 1),
            nn.LeakyReLU(),
            nn.Softmax(dim=2),
        )

        self.normalized_activation = nn.Sequential(
            nn.LayerNorm(self.bond_fea_len),
            nn.Softplus(),
        )

    def forward(
        self,
        bond_features: torch.Tensor,
        angle_features: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the BondConv module.

        Args:
            bond_features (torch.Tensor): bond features of shape `(n_at * k, n_radial_bond)`.
            angle_features (torch.Tensor): angle features of shape
                `(n_at * k, k - 1, n_radial_angle)`.
            neighbor_indices (torch.Tensor): neighbor indices, shape `(n_at * k, k - 1)`.

        Returns:
            torch.Tensor: The output of the module, of shape `(n_at * k, n_radial_bond)`.
        """
        n = neighbor_indices.size(1)  # k - 1
        m = neighbor_indices.size(0)  # n_at * k

        # (n_at * k, k - 1, n_radial_bond)
        eij = bond_features.unsqueeze(1).expand(m, n, self.bond_fea_len)
        eik = bond_features[neighbor_indices]

        # (n_at * k, k - 1, 2 * n_radial_bond + n_radial_angle)
        cat_fea = torch.cat([eij, eik, angle_features], dim=2)

        # (n_at * k, n_radial_bond)
        output = self.normalized_activation(
            bond_features
            + torch.sum(
                self.normalized_activation(self.attention(cat_fea) * self.linear(cat_fea)),
                dim=1,
            )
        )

        return output
