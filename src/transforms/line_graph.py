import torch
from line_profiler import profile
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import scatter


def compute_bonds_angles(x: Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Computes bond angles (in radians) for all *directed* neighbor pairs of bonds.

    Args:
        x (torch.Tensor): Displacement components between central atom i and neighbors j
            and k in all 3 spatial dimensions for all unordered neighbor triplets in the graph.
        eps (float): A small value to avoid division by zero.

    Returns:
        torch.Tensor: Angles (in radians) for all *directed* neighbor pairs of bonds
            (i → j, i → k) and (i → k, i → j), shape (2 * num_triplets, 1).
    """
    # For each unordered pair (j, k), we build two directed pairs:
    #   (i -> j, i -> k) and (i -> k, i -> j)
    v1 = torch.cat((x[:, :3], x[:, 3:]), dim=0)  # first bond in the pair
    v2 = torch.cat((x[:, 3:], x[:, :3]), dim=0)  # second bond in the pair

    # Cosine of the angle between v1 and v2
    denom = v1.norm(dim=1) * v2.norm(dim=1) + eps
    cos_theta = (v1 * v2).sum(dim=1) / denom
    cos_theta = cos_theta.clamp(-1.0, 1.0)

    angles = torch.acos(cos_theta)

    return angles


class LineGraphData(Data):
    """Custom Data class for LineGraph to handle batching of bond-to-atom indices."""

    def __inc__(self, key: str, value: torch.Tensor, *args, **kwargs) -> int:
        if key in ["bond_source", "bond_target"]:
            return self.num_atoms
        return super().__inc__(key, value, *args, **kwargs)


# TODO consider computing triplets in model forward instead
# TODO fix angle flow direction (currently j -> i -> k instead of k -> j -> i)
class LineGraph(BaseTransform):
    """Converts a graph to its directed line-graph version.

    The implementation is based on `torch_geometric.transforms.LineGraph` with three key
    differences:

    1. We assume the graph is directed, meaning the resulting line-graph will be directed.
        It is equivalent to setting `force_directed=True` in the original implementation.

    2. We set `edge_attr` to be the cosine of the angle between bonds.

    3. We avoid coalescing the graph to ensure periodicity invariance.
    """

    # TODO try to optimize
    @profile
    def _get_new_adj(
        self, old_row: Tensor, old_col: Tensor, num_atoms: int, num_bonds: int
    ) -> tuple[list[Tensor], list[Tensor]]:
        device = old_row.device

        i = torch.arange(num_bonds, dtype=torch.long, device=device)

        # We want k-1 edges to avoid angles between a bond and itself
        count = (
            scatter(
                src=torch.ones_like(old_row),
                index=old_row,
                dim=0,
                dim_size=num_atoms,
                reduce="sum",
            )
            - 1
        )

        # build ptr as CSR-style pointer: size = num_atoms + 1
        ptr = torch.empty(num_atoms + 1, dtype=torch.long, device=device)
        ptr[0] = 0
        ptr[1:] = count.cumsum(dim=0)

        # Precompute the slice for each atom
        atom_cols: list[Tensor] = [i[ptr[a] : ptr[a + 1]] for a in range(num_atoms)]

        cols: list[Tensor] = [atom_cols[v.item()] for v in old_col]  # type: ignore
        rows: list[Tensor] = [old_row.new_full((c.numel(),), j) for j, c in enumerate(cols)]

        return rows, cols

    @profile
    def forward(self, data: Data) -> Data:
        """Modified Linegraph forward but also adds cosine angles as LineGraph edge_attr. The
        resulting graph will be directed.

        Args:
            data (Data): the PyG Data graph to be converted into a LineGraph.

        Returns:
            data (Data): a LineGraph data object.
        """
        assert data.edge_index is not None
        assert data.x is not None

        # Original graph data
        edge_index = data.edge_index
        row, col = edge_index

        num_atoms = data.num_nodes
        num_bonds = edge_index.size(1)

        # Compute angle cosines (line graph edge attributes)
        new_edge_attr = compute_bonds_angles(data.x)

        # Each bond j has a central atom col[j]
        bond_source = row.clone()
        bond_target = col.clone()

        # New adjacency for the line graph
        rows, cols = self._get_new_adj(row, col, num_atoms, num_bonds)

        lg_row, lg_col = torch.cat(rows, dim=0), torch.cat(cols, dim=0)
        lg_edge_index = torch.stack([lg_row, lg_col], dim=0)

        # Update data object (nodes are now bonds, edges are angles)
        data.x = data.edge_attr
        data.edge_attr = new_edge_attr
        data.edge_index = lg_edge_index
        data.num_nodes = num_bonds

        # Store additional metadata
        data.bond_source = bond_source
        data.bond_target = bond_target
        data.num_atoms = num_atoms

        return LineGraphData(**data.to_dict())
