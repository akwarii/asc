import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import cumsum, scatter


def compute_bonds_cosines(x: torch.Tensor, n_bonds: int, eps: float = 1e-8) -> torch.Tensor:
    """Computes the bond angle cosines from the bond displacement vectors for all triplets
    in the graph.

    Args:
        x (torch.Tensor): A tensor with distances components between central atom i and
            either neighbor j (x[:,:3]) or k (x[:,3:]) in all 3 spatial dimensions for all
            neighbor triplets in the graph.
        n_bonds (int): The number of edges in the graph.
        eps (float): A small value to avoid division by zero.

    Returns:
        torch.Tensor: angle cosines for all neighbor triplets in the graph.
    """
    k = (2 * x.size(0)) // n_bonds  #  this is actually "k - 1" with k nearest neighbors
    v1 = torch.cat((x[:, :3], x[:, 3:])).reshape(n_bonds, k, 3)  # (i -> j, i -> k)
    v2 = torch.cat((x[:, 3:], x[:, :3])).reshape(n_bonds, k, 3)  # (i -> k, i -> j)
    angle_cos = (v1 * v2).sum(dim=2) / (v1.norm(dim=2) * v2.norm(dim=2) + eps)
    return angle_cos.flatten().unsqueeze(1)


class LineGraph(BaseTransform):
    """Converts a graph to its directed line-graph version.

    The implementation is bases on `torch_geometric.transforms.LineGraph` with three key
    differences:

    1. We assume the graph is directed, meaning the resulting line-graph will be directed.
        It is equivalent to setting `force_directed=True` in the original implementation.

    2. We set `edge_attr` to be the cosine of the angle between bonds.

    3. We avoid coalescing the graph to ensure periodicity invariance.
    """

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
        new_edge_attr = compute_bonds_cosines(data.x, num_bonds)

        # Each bond j has a central atom col[j]
        bond_source = row.clone()
        bond_target = col.clone()

        i = torch.arange(num_bonds, dtype=torch.long, device=row.device)

        # We want k-1 edges to avoid angles between a bond and itself
        count = (
            scatter(
                src=torch.ones_like(row),
                index=row,
                dim=0,
                dim_size=num_atoms,
                reduce="sum",
            )
            - 1
        )
        ptr = cumsum(count)

        cols = [i[ptr[col[j]] : ptr[col[j] + 1]] for j in range(col.size(0))]
        rows = [row.new_full((c.numel(),), j) for j, c in enumerate(cols)]

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

        return data
