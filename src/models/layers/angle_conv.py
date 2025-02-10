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

        self.norm = nn.LayerNorm(self.bond_fea_len)
        self.act = nn.Softplus()

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        neigh_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the AngleConv module.

        Args:
            x (torch.Tensor): bond features of shape `(n_at * k, n_radial_bond)`.
            edge_attr (torch.Tensor): angle features of shape `(n_at * k, k - 1, n_radial_angle)`.
            neigh_index (torch.Tensor): neighbor indices, shape `(n_at * k, k - 1)`.

        Returns:
            torch.Tensor: The output of the module, of shape `(n_at * k, k - 1, n_radial_angle)`.
        """
        m = neigh_index.size(0)  # N_at * k
        n = neigh_index.size(1)  # k - 1

        # (n_at * k, k - 1, n_radial_bond)
        eij = x.unsqueeze(1).expand(m, n, self.bond_fea_len)
        eik = x[neigh_index]

        # (n_at * k, k - 1, 2 * n_radial_bond)
        eijk = torch.cat([eij, eik], dim=2)
        aijk = edge_attr.view(m, n, self.angle_fea_len)

        # (n_at * k, k - 1, 2 * n_radial_bond + n_radial_angle)
        features = torch.cat([eijk, aijk], dim=2)

        x_att = self.attention(features) * self.linear(features)
        x_att = self.act(x_att)

        # output: (n_at * k, k - 1, n_radial_angle)
        output = aijk + x_att
        output = self.norm(output)
        output = self.act(output)

        return output
