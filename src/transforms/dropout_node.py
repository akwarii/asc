import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import dropout_node


class DropoutNode(BaseTransform):
    """Randomly drops nodes from a graph.

    Uses `torch_geometric.utils.dropout_node` to drop each node with
    probability `rate`. The transform itself is applied with probability `p`.

    Args:
        rate (float): Per-node drop probability in [0., 1.[.
        seed (int): Random seed used to decide whether to apply the transform.
        p (float): Probability to apply the transform on a given sample.
    """

    def __init__(
        self, rate: float = 0.05, seed: int = 42, p: float = 0.1,
    ) -> None:
        if not (0.0 <= rate < 1.0):
            raise ValueError("rate must be in [0., 1.[")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0., 1.[")

        super().__init__()

        # ? Note : I think it is better to have two distinct probabilities
        # ?        one for applying the transform and one for dropping nodes.
        # ?        I dont know how suitable the default values are though.
        self.rate = rate
        self.p = p
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

    # ? Note : if we want to use it as data augmentation, it means we directly modify
    # ?        the input data.
    # ? Also, as augmentation is applied during training only, we hard-set `training=True`
    # ? when calling pyg dropout_node.
    def forward(self, data: Data) -> Data:
        """Applies random nodes dropout to a graph.

        Args:
            data (Data): The input graph data.

        Returns:
            Data: The graph data with nodes dropped.
        """
        # Apply transform/augmentations with probability p
        if torch.rand(1, generator=self.rng).item() > self.p:
            return data

        # Use PyG's dropout_node to compute new edges and mask
        edge_index, edge_mask, node_mask = dropout_node(
            data.edge_index,
            p=self.rate,
            num_nodes=data.num_nodes,
            training=True,
        )

        # If all nodes were dropped, keep original to avoid empty graphs
        num_nodes = node_mask.sum().item()
        if num_nodes == 0:
            return data
        data.num_nodes = num_nodes

        data.x = data.x[node_mask]
        data.edge_index = edge_index

        # Propagate mask to standard and custom edge attributes if present
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[edge_mask]

        # As this transform is meant to be used on line-graphs, the mask should also
        # be applied to other features.
        # ! @Gael, can you check that I forgot nothing among the attributes you use?
        if hasattr(data, "num_atoms"):  # we have a line-graph (attribute added in line_graph.py)
            keys_to_check = ["bond_source", "bond_target"]
        else:
            keys_to_check = []  # no extra attributes to consider for standard graphs
        for key in keys_to_check:
            if hasattr(data, key) and data[key] is not None:
                data[key] = data[key][node_mask]

        return data
