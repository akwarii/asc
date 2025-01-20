import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class RandomPerturbation(BaseTransform):
    """Applies random Gaussian noise to both node and edge features of a graph if they exist.

    Args:
        std: Standard deviation of the applied noise.
    """

    def __init__(self, std: float = 0.01) -> None:
        if std <= 0.0:
            raise ValueError("The standard deviation must be strictly positive.")

        self.stddev = std

    def forward(self, data: Data) -> Data:
        """Runs the transform."""
        if data.x is not None:
            data.x = data.x + torch.normal(mean=0.0, std=self.stddev, size=data.x.size())

        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr + torch.normal(
                mean=0.0, std=self.stddev, size=data.edge_attr.size()
            )

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(stddev={self.stddev})"
