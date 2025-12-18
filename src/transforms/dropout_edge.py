import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import dropout_edge


class DropoutEdge(BaseTransform):
    """Randomly drops edges from a graph.

    Uses `torch_geometric.utils.dropout_edge` to drop each edge with
    probability `rate`. The transform itself is applied with probability `p`.

    Args:
        rate (float): Per-edge drop probability in [0., 1.[.
        seed (int): Random seed used to decide whether to apply the transform.
        p (float): Probability to apply the transform on a given sample.
        force_undirected (bool): If True, ensures edges are dropped in pairs
            for undirected graphs. Defaults to False.
    """

    def __init__(
        self, rate: float = 0.05, seed: int = 42, p: float = 0.1, *, force_undirected: bool = False
    ) -> None:
        if not (0.0 <= rate < 1.0):
            raise ValueError("rate must be in [0., 1.[")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0., 1.[")

        super().__init__()

        # ? Note : I think it is better to have two distinct probabilities
        # ?        one for applying the transform and one for dropping edges.
        # ?        I dont know how suitable the default values are though.
        self.rate = rate
        self.p = p
        self.force_undirected = force_undirected  # ! with KNN graphs this should (always?) be True
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

    # ? Note : if we want to use it as data augmentation, it means we directly modify
    # ?        the input data.
    # ? Also, as augmentation is applied during training only, we hard-set `training=True`
    # ? when calling pyg dropout_edge.
    def forward(self, data: Data) -> Data:
        """Applies random edge dropout to a graph.

        Args:
            data (Data): The input graph data.

        Returns:
            Data: The graph data with edges dropped.
        """
        if data.edge_index is None:
            return data

        # Apply transform/augmentations with probability p
        if torch.rand(1, generator=self.rng).item() > self.p:
            return data

        # Use PyG's dropout_edge to compute new edges and mask
        edge_index, edge_mask = dropout_edge(
            data.edge_index,
            p=self.rate,
            force_undirected=self.force_undirected,
            training=True,
        )

        # If all edges were dropped, keep original to avoid empty graphs
        if edge_index.size(1) == 0:
            return data
        # ? could also consider re-sampling until at least one edge remains ?
        # ? what about checking that each node has at least one edge ?

        data.edge_index = edge_index

        # Propagate mask to standard and custom edge attributes if present
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[edge_mask]

        # As this transform is meant to be used on line-graphs, the mask should also
        # be applied to other features.
        # ! @Gael, can you check that I forgot nothing among the attributes you use?
        if hasattr(data, "num_nodes"):  # we have a line-graph
            keys_to_check = ["bond_source", "bond_target"]
        else:
            keys_to_check = []  # no extra attributes to consider for standard graphs
        for key in keys_to_check:
            if hasattr(data, key) and data[key] is not None:
                data[key] = data[key][edge_mask]

        return data
