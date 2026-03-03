import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import cumsum, scatter


def compute_bonds_angles(x: Tensor, lg_edge_index: Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Computes bond angles for all *directed* neighbor pairs of bonds.

    Args:
        x (torch.Tensor): Distance vector between atom i and j in all 3 spatial dimensions.
        lg_edge_index (torch.Tensor): The edge indices of the line graph, shape (2, num_lg_edges).
        eps (float): A small value to avoid division by zero.

    Returns:
        torch.Tensor: Angle cosines for all *directed* neighbor pairs of bonds
            shape (num_lg_edges, 1).
    """
    v1 = x[lg_edge_index[0]]
    v2 = x[lg_edge_index[1]]

    dot_product = (v1 * v2).sum(dim=-1)
    norm_v1 = v1.norm(dim=-1)
    norm_v2 = v2.norm(dim=-1)

    cos_theta = dot_product / (norm_v1 * norm_v2 + eps)
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)

    return cos_theta.unsqueeze(-1)


# TODO check if we can "easily" ensure that num_atoms stays an int after batching
class LineGraphData(Data):
    """Custom Data class for LineGraph to handle batching of bond-to-atom indices."""

    def __inc__(self, key: str, value: torch.Tensor, *args, **kwargs) -> int:
        if key in ["bond_source", "bond_target"]:
            if hasattr(self, "num_atoms"):
                return self.num_atoms
            else:
                raise AttributeError(
                    "LineGraphData object is missing 'num_atoms' attribute required for batching."
                )
        return super().__inc__(key, value, *args, **kwargs)


class LineGraph(BaseTransform):
    """Converts a graph to its directed line-graph version.

    The implementation is based on `torch_geometric.transforms.LineGraph` with three key
    differences:

    1. We assume the graph is directed, meaning the resulting line-graph will be directed.
        It is equivalent to setting `force_directed=True` in the original implementation.

    2. We set `edge_attr` to be the cosine of the angle between bonds.

    3. We avoid coalescing the graph to ensure periodicity invariance.
    """

    def forward(self, data: Data) -> Data:
        """An optimized version of the LineGraph forward method that avoids coalescing and
        directly computes the new edge_index without intermediate list constructions. This is
        expected to be much faster for large graphs.

        Note: This method is not fully implemented yet and serves as a placeholder for the
        optimized logic.
        """
        # Ensure edge indices are sorted
        data = data.sort(sort_by_row=True)

        assert data.edge_index is not None
        assert data.edge_attr is not None
        assert data.num_nodes is not None
        assert data.x is not None

        # Original graph data
        edge_index, edge_attr = data.edge_index, data.edge_attr
        row, col = edge_index

        num_atoms = data.num_nodes
        num_edges = data.num_edges

        # Each bond j has a central atom col[j]
        bond_source = row.clone()
        bond_target = col.clone()

        # Compute the directed line graph adjacency
        # The implementation is similar to PyG (without coalesce) but much more efficient
        # since we avoid nested loops and use PyTorch operations directly on tensors.
        count = scatter(torch.ones_like(row), row, dim=0, dim_size=num_atoms, reduce="sum")
        ptr = cumsum(count)

        # Determine how many outgoing bonds each target atom has
        repeats = count[col]

        # Repeats the index of bond 'j' for every outgoing bond from its target atom
        lg_row = torch.repeat_interleave(torch.arange(num_edges, device=row.device), repeats)

        total_lg_edges = repeats.sum().item()

        # Generate a local index (0, 1, 2... for each group)
        cum_repeats = repeats.cumsum(0)
        offsets = torch.arange(total_lg_edges, device=row.device) - torch.repeat_interleave(
            cum_repeats - repeats, repeats
        )
        # Add the start pointer for each target atom and add the offset
        lg_col = ptr[col].repeat_interleave(repeats) + offsets

        # Stack to get the new edge_index for the line graph
        new_edge_index = torch.stack([lg_row, lg_col], dim=0)

        # Compute angle cosines (line graph edge attributes)
        new_edge_attr = compute_bonds_angles(edge_attr, new_edge_index)

        # Compute distance magnitudes (line graph node attributes)
        new_node_attr = edge_attr.norm(dim=-1, keepdim=True)

        # Update data object (nodes are now bonds, edges are angles)
        data.x = new_node_attr
        data.edge_attr = new_edge_attr
        data.edge_index = new_edge_index
        data.num_nodes = num_edges

        # Store additional metadata
        data.bond_source = bond_source
        data.bond_target = bond_target
        data.num_atoms = num_atoms

        return LineGraphData(**data.to_dict())
