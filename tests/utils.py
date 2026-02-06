from dataclasses import dataclass

import torch
from torch import Tensor
from torch_geometric.data import Data

__all__ = ["MockGraphSpec", "make_mock_pyg_graph"]

@dataclass(frozen=True)
class MockGraphSpec:
    num_nodes: int
    num_edges: int
    node_feat_dim: int = 8
    edge_feat_dim: int = 16
    directed: bool = True
    allow_self_loops: bool = True
    allow_duplicate_edges: bool = True
    seed: int = 42
    device: str = "cpu"


def make_mock_pyg_graph(spec: MockGraphSpec) -> Data:
    """
    Create a PyG Data graph suitable for unit tests.

    - edge_index is [2, E] (or [2, 2E] if undirected=True)
    - x is [N, node_feat_dim] if node_feat_dim>0
    - edge_attr is [E, edge_feat_dim] (or [2E, edge_feat_dim] if undirected=True)
    """
    if spec.num_nodes <= 0:
        raise ValueError("num_nodes must be > 0")
    if spec.num_edges <= 0:
        raise ValueError("num_edges must be > 0")

    device = spec.device

    rng = torch.Generator(device=device)
    if spec.seed is not None:
        rng.manual_seed(int(spec.seed))
    else:
        rng.seed()

    N = spec.num_nodes
    E = spec.num_edges

    # ---- Build edges ----
    if spec.allow_duplicate_edges:
        row = torch.randint(0, N, (E,), generator=rng, device=device, dtype=torch.long)
        col = torch.randint(0, N, (E,), generator=rng, device=device, dtype=torch.long)
    else:
        # sample unique pairs from the cartesian product
        total_pairs = N * N if spec.allow_self_loops else N * (N - 1)
        if total_pairs == 0:
            raise ValueError("Cannot sample edges with num_nodes=1 and allow_self_loops=False")

        if E > total_pairs:
            raise ValueError(
                f"Requested {E} unique edges but only {total_pairs} possible "
                f"(num_nodes={N}, allow_self_loops={spec.allow_self_loops})."
            )

        # map k in [0,total_pairs) to (i,j)
        perm = torch.randperm(total_pairs, generator=rng, device=device)[:E]

        if spec.allow_self_loops:
            row = perm // N
            col = perm % N
        else:
            # pairs over i != j; index i*(N-1) + j', where j' skips i
            row = perm // (N - 1)
            jprime = perm % (N - 1)
            col = jprime + (jprime >= row).to(torch.long)

    if not spec.allow_self_loops:
        mask = row != col
        if mask.sum().item() != E:
            # This should not happen in the unique sampling path, but keep robust.
            row, col = row[mask], col[mask]

    edge_index = torch.stack([row, col], dim=0).to(device)

    # Make undirected by adding reverse edges (and duplicating edge_attr later)
    if not spec.directed:
        rev = edge_index.flip(0)
        edge_index = torch.cat([edge_index, rev], dim=1)

    # ---- Node features ----
    x: Tensor | None
    if spec.node_feat_dim > 0:
        x = torch.randn((N, spec.node_feat_dim), generator=rng, device=device)
    else:
        x = None

    # ---- Edge features ----
    edge_attr: Tensor | None
    if spec.edge_feat_dim > 0:
        edge_attr = torch.randn(
            (edge_index.size(1), spec.edge_feat_dim), generator=rng, device=device
        )
    else:
        edge_attr = None

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=N)

    data.validate(raise_on_error=True)
    return data
