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

        self.norm = nn.LayerNorm(self.bond_fea_len)
        self.act = nn.Softplus()

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        neigh_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the BondConv module.

        Args:
            x (torch.Tensor): bond features of shape `(n_at * k, n_radial_bond)`.
            edge_attr (torch.Tensor): angle features of shape `(n_at * k, k - 1, n_radial_angle)`.
            neigh_index (torch.Tensor): neighbor indices, shape `(n_at * k, k - 1)`.

        Returns:
            torch.Tensor: The output of the module, of shape `(n_at * k, n_radial_bond)`.
        """
        m = neigh_index.size(0)  # N_at * k
        n = neigh_index.size(1)  # k - 1

        # (n_at * k, k - 1, n_radial_bond)
        eij = x.unsqueeze(1).expand(m, n, self.bond_fea_len)
        eik = x[neigh_index]
        aijk = edge_attr.view(m, n, self.angle_fea_len)

        # (n_at * k, k - 1, 2 * n_radial_bond + n_radial_angle)
        features = torch.cat([eij, eik, aijk], dim=2)

        x_att = self.attention(features) * self.linear(features)
        x_att = self.norm(x_att)
        x_att = self.act(x_att)

        # (n_at * k, n_radial_bond)
        output = x + torch.sum(x_att, dim=1)
        output = self.norm(output)
        output = self.act(output)

        return output
