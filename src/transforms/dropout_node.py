from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import dropout_node


class DropoutNode(BaseTransform):
    """Randomly drops nodes from a graph. Each node is dropped with
    probability p using samples from a Bernoulli distribution.

    This is a transform version of `torch_geometric.utils.dropout_node`.

    Note:
        This transform being non-deterministic, it is intended to be used only during training. It
        is therefore automatically removed from the validation, testing and prediction dataloaders
        when created by the `LightningDataset`.

    Args:
        p (float): Dropout probability (default: 0.1).
    """

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        self.p = p

    def forward(self, data: Data) -> Data:
        """Applies node dropout to a graph.

        Args:
            data (Data): The input graph data.

        Returns:
            Data: The transformed graph.
        """
        assert data.edge_index is not None

        # Use PyG's dropout_node to compute new edges and mask
        _, _, node_mask = dropout_node(
            data.edge_index,
            p=self.p,
            num_nodes=data.num_nodes,
            training=True,
        )

        new_data = data.subgraph(node_mask)

        if new_data.num_nodes == 0:
            new_data = data  # Avoid empty graph

        return new_data
