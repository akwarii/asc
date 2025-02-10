import torch
from torch_geometric.data import Data
from torch_geometric.nn import MLP as PyGMLP  # noqa

from src.models.expansion import GaussianBasis


class MLP(PyGMLP):  # noqa
    def __init__(self, num_radial: int, *args, **kwargs) -> None:  # noqa
        super().__init__(*args, **kwargs)

        self.rbf = GaussianBasis(num_radial=num_radial)
        self.sbf = GaussianBasis(num_radial=num_radial)

        self.lin_edge = torch.nn.Linear(num_radial, self.out_channels, bias=False)
        self.lin_node = torch.nn.Linear(num_radial, self.out_channels)

    def forward(self, data: Data) -> torch.Tensor:  # noqa
        assert data.x is not None
        assert data.edge_attr is not None
        assert data.edge_index is not None
        assert data.num_nodes is not None

        x, edge_attr = data.x, data.edge_attr

        num_nodes, num_edges = data.num_nodes, edge_attr.size(0)
        num_radial = self.rbf.num_radial
        k = num_edges // num_nodes + 1
        num_atoms = num_nodes // k

        # Embed the features
        x: torch.Tensor = self.rbf(x)
        edge_attr: torch.Tensor = self.sbf(edge_attr)

        # Aggregate the features and reshape the tensor to the inverse line graph structure
        edge_attr = edge_attr.view(num_nodes, (k - 1) * num_radial)
        features = torch.cat([x, edge_attr], dim=1)
        features = features.view(num_atoms, k, -1).flatten(1)

        out = super().forward(features)

        return out
