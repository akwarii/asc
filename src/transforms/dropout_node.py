import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import dropout_node


class DropoutNode(BaseTransform):
    """Randomly drops nodes from a graph.

    Uses `torch_geometric.utils.dropout_node` to drop each node with
    probability `rate`. The transform itself is applied with probability `p`.
    The two probabilities serve different purposes, controlling the likelihood
    of creating vacancies and the extent of vacancies in the graph, respectively.

    Args:
        rate (float): Per-node drop probability in [0., 1.[.
        seed (int): Random seed used to decide whether to apply the transform.
        p (float): Probability to apply the transform on a given sample.
    """

    def __init__(
        self,
        rate: float = 0.05,
        p: float = 0.1,
        seed: int = 42,
    ) -> None:
        if not (0.0 <= rate < 1.0):
            raise ValueError("rate must be in [0., 1.[")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0., 1.]")

        super().__init__()
        self.rate = rate
        self.p = p
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

    def forward(self, data: Data) -> Data:
        """Applies random nodes dropout to a graph.

        Args:
            data (Data): The input graph data.

        Returns:
            Data: The graph data with nodes dropped.
        """
        assert data.edge_index is not None

        # Apply transform/augmentations with probability p
        if torch.rand(1, generator=self.rng).item() > self.p:
            return data

        # Use PyG's dropout_node to compute new edges and mask
        _, _, node_mask = dropout_node(
            data.edge_index,
            p=self.rate,
            num_nodes=data.num_nodes,
            training=True,
        )

        new_data = data.subgraph(node_mask)

        if new_data.num_nodes == 0:
            new_data = data  # Avoid empty graph

        return new_data
