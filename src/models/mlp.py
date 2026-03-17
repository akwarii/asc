import torch
from torch import Tensor
from torch_geometric.nn import MLP

from src.models.base import BaseModel
from src.models.expansion import GaussianBasis


class MLPClassifier(BaseModel, MLP):  # noqa
    def __init__(self, num_radial: int, *args, **kwargs) -> None:  # noqa
        MLP.__init__(self, *args, **kwargs)

        self.rbf = GaussianBasis(num_radial=num_radial)
        self.sbf = GaussianBasis(num_radial=num_radial)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:  # type: ignore
        """Forward pass of the model.

        Args:
            x (Tensor): The node features.
            edge_index (Tensor): The neighbor indices.
            edge_attr (Tensor): The edge features.
        """
        num_nodes, num_edges = x.size(0), edge_attr.size(0)
        num_radial = self.rbf.num_radial
        k = num_edges // num_nodes + 1
        num_atoms = num_nodes // k

        # Embed the features
        x_emb: Tensor = self.rbf(x)
        edge_attr_emb: Tensor = self.sbf(edge_attr)

        # Aggregate the features and reshape the tensor to the inverse line graph structure
        edge_attr_emb = edge_attr_emb.view(num_nodes, (k - 1) * num_radial)
        features = torch.cat([x_emb, edge_attr_emb], dim=1)
        features = features.view(num_atoms, k, -1).flatten(1)

        out = super().forward(features)

        return out
