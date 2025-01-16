import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


# FIXME: this class is not used in the current implementation
# as it will make the code crash. We need to move to the usual PyG
# Data storage of the targets and use the PyG collate function.
class DropoutNode(BaseTransform):
    """Class to apply random node dropout to boxes.

    Randomly drops every node of a graph with probability p.
    This mimics crystalline defects in units cells (without
    relaxation).

    Args:
        rate (float): dropout probability in [0.,1.] (default=0.05).
        seed (int): Random seed (default=42).
        p (float): probability to apply the transform (default=0.1).
    """

    def __init__(self, rate: float = 0.05, seed: int = 42, p: float = 0.1) -> None:
        if rate == 1.0:
            raise ValueError("The dropout rate must be strictly less than 1.0.")

        if p < 0.0 or p > 1.0:
            raise ValueError("The dropout probability must be in [0.,1.].")

        super().__init__()

        self.rate = rate
        self.p = p
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

    def forward(self, x: Data) -> Data:
        """Applies random node dropout to a batch of graphs."""
        # TODO we may switch to modify the Data object every time but with a given probability
        if torch.rand(1, generator=self.rng).item() > self.p:
            return x

        if x.num_nodes is None:
            raise ValueError("The number of nodes must be provided.")

        num_nodes_kept = 0
        while num_nodes_kept == 0:
            node_mask = torch.rand(x.num_nodes, generator=self.rng) > self.rate
            num_nodes_kept = torch.sum(node_mask).item()

        # Updated nodes/positions
        if x.pos is not None:
            pos = x.pos[node_mask]

        # Need to remove edges involving the dropped nodes
        if x.edge_index is not None:
            edge_mask = node_mask[x.edge_index[0]]
            edge_index = x.edge_index[:, edge_mask]

        edge_dist = x.edge_dist[edge_mask] if edge_mask is not None else None
        angle_cos = x.angle_cos[edge_mask] if edge_mask is not None else None

        augmented_graph = Data(
            num_nodes=num_nodes_kept,
            pos=pos,
            edge_index=edge_index,
            edge_dist=edge_dist,
            angle_cos=angle_cos,
        )

        return augmented_graph

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(rate={self.rate}, p={self.p})"
