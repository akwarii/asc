from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import dropout_edge


class DropoutEdge(BaseTransform):
    """Randomly drops edges from the adjacency matrix edge_index with
    probability p using samples from a Bernoulli distribution.

    This is a transform version of `torch_geometric.utils.dropout_edge`.

    Note:
        This transform being non-deterministic, it is intended to be used only during training. It
        is therefore automatically removed from the validation, testing and prediction dataloaders
        when created by the `LightningDataset`.

    Args:
        p (float): Dropout probability (default: 0.1).
        force_undirected (bool): If set to True, will either drop or keep both
            edges of an undirected edge. (default: True)
    """

    def __init__(self, p: float = 0.1, *, force_undirected: bool = True) -> None:
        super().__init__()

        self.p = p
        self.force_undirected = force_undirected

    def forward(self, data: Data) -> Data:
        """Applies edge dropout to a graph.

        Args:
            data (Data): The input graph data.

        Returns:
            Data: The transformed graph.
        """
        assert data.edge_index is not None

        # Use PyG's dropout_edge to compute new edges and mask
        edge_index, edge_mask = dropout_edge(
            data.edge_index,
            p=self.p,
            force_undirected=self.force_undirected,
            training=True,
        )

        # If all edges were dropped, keep original to avoid empty graphs
        if edge_index.size(1) == 0:
            return data

        data.edge_index = edge_index

        # Propagate mask to standard and custom edge attributes if present
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[edge_mask]

        return data
