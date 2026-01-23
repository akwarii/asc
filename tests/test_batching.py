import time

import torch
from src.models.layers.readout import BondToAtomReadout
from src.transforms.line_graph import LineGraph
from torch_geometric.data import Batch, Data


def create_dummy_graph(num_atoms=5, num_neighbors=2):
    """Creates a dummy graph with random features."""
    # Create a simple cycle graph or random graph
    # x: atom positions/features? In KNNGraph it is (k_coords - central_coords) || (j_coords - central_coords)
    # But for LineGraph transform input, 'x' is expected to be edge vectors (dim 3+3=6 from code analysis)
    # Wait, in KNNGraph.convert:
    # x = cat(j-central, k-central) -> shape (num_edges, 6)
    # edge_index -> shape (2, num_edges)
    # So input to LineGraph has 'x' as bond features (geometric vectors)

    num_bonds = num_atoms * num_neighbors
    x = torch.randn(num_bonds, 6)

    # Create random edge_index
    # row (source/central atom), col (neighbor)
    row = torch.arange(num_atoms).repeat_interleave(num_neighbors)
    col = torch.randint(0, num_atoms, (num_bonds,))
    edge_index = torch.stack([row, col], dim=0)

    data = Data(x=x, edge_index=edge_index, num_nodes=num_atoms)
    return data


def test_line_graph_transform_integrity():
    """Test that LineGraph transform produces expected structure and attributes."""
    num_atoms = 10
    k = 4
    data = create_dummy_graph(num_atoms=num_atoms, num_neighbors=k)
    data.edge_attr = torch.randn(data.x.size(0), 1)

    transform = LineGraph()
    lg_data = transform(data)

    # Check attributes exist
    assert hasattr(lg_data, "bond_source")
    assert hasattr(lg_data, "bond_target")
    assert hasattr(lg_data, "num_atoms")

    # Check dimensions
    num_bonds = data.x.size(0)
    assert lg_data.num_nodes == num_bonds
    assert lg_data.x.shape == (num_bonds, 1)  # edge_attr (distances) became x, but wait.
    # In LineGraph.forward:
    # data.x = data.edge_attr
    # But input data from create_dummy_graph didn't have edge_attr set.
    # KNNGraph sets edge_attr = distances.
    # Let's fix dummy data to have edge_attr


def test_batching_indices():
    """Test if bond_source is correctly incremented when batching."""
    transform = LineGraph()

    # Graph 1
    g1 = create_dummy_graph(num_atoms=10, num_neighbors=3)
    g1.edge_attr = torch.randn(g1.x.size(0), 1)  # Dummy distances
    lg1 = transform(g1)

    # Graph 2
    g2 = create_dummy_graph(num_atoms=20, num_neighbors=3)
    g2.edge_attr = torch.randn(g2.x.size(0), 1)
    lg2 = transform(g2)

    # Batch them
    batch = Batch.from_data_list([lg1, lg2])

    # Check sizes
    total_bonds = lg1.num_nodes + lg2.num_nodes
    assert batch.num_nodes == total_bonds
    assert batch.bond_source.size(0) == total_bonds

    # Check bond_source values
    # lg1.bond_source should be in [0, 9]
    # lg2.bond_source should be in [0, 19]
    # batch.bond_source for the second graph should be shifted by lg1.num_atoms (10)

    # Indices for the second graph in the batch
    batch_slice = batch.batch == 1
    offset_bond_source = batch.bond_source[batch_slice]

    expected_min = 10  # num_atoms of g1
    expected_max = 10 + 20 - 1

    print(f"Batching Check:")
    print(f"  G1 atoms: {g1.num_nodes}, G2 atoms: {g2.num_nodes}")
    print(
        f"  Batch bond_source min/max (2nd graph): {offset_bond_source.min().item()}/{offset_bond_source.max().item()}"
    )

    # This assertion is expected to fail if __inc__ is not correctly implemented
    assert offset_bond_source.min() >= expected_min, "bond_source was not incremented correctly!"
    assert offset_bond_source.max() <= expected_max


def test_readout_on_batch():
    """Test BondToAtomReadout on a batch."""
    transform = LineGraph()
    g1 = create_dummy_graph(num_atoms=5, num_neighbors=2)
    g1.edge_attr = torch.randn(g1.x.size(0), 1)
    lg1 = transform(g1)

    g2 = create_dummy_graph(num_atoms=5, num_neighbors=2)
    g2.edge_attr = torch.randn(g2.x.size(0), 1)
    lg2 = transform(g2)

    batch = Batch.from_data_list([lg1, lg2])

    # Fake embedding output (batch_num_nodes, channels)
    x_emb = torch.randn(batch.num_nodes, 16)

    readout = BondToAtomReadout(reduce="mean", incidence="out")

    # This requires bond_source to be correctly set/incremented
    # and num_atoms to be the total number of atoms in the batch

    # PyG Batch object usually doesn't sum 'num_atoms' attribute automatically unless it's standard.
    # Actually Batch.num_nodes is sum of num_nodes.
    # But here 'num_nodes' in LineGraph is 'num_bonds'.
    # We need total atoms for scatter dimension.

    # We expect 10 atoms total (5 from g1, 5 from g2)
    expected_total_atoms = 10

    # Force the correct total_atoms to test if readout handles the indices provided by batch
    # If batch.bond_source is wrong (not incremented), it will be [0..4, 0..4]
    # Then scatter will only fill 0..4.

    # We use the batch properties to derive total_atoms mimicking real usage
    # But for the test assertion we know it must be 10.

    # Let's try to infer what the model would see.
    # The model usually relies on batch.num_atoms or max index.

    # Check if bond_source covers the expected range
    if batch.bond_source.max().item() < 5:
        print("  WARNING: bond_source seems NOT incremented in batch!")

    # Perform readout
    # We pass expected_total_atoms to ensure output tensor is large enough to check for zeros
    out = readout(x_emb, expected_total_atoms, bond_source=batch.bond_source)

    assert out.size(0) == expected_total_atoms
    assert out.size(1) == 16

    # Check if the second half of the output is not all zeros (assuming embeddings are random/non-zero)
    # If bond_source was [0..4, 0..4], then indices 5..9 in 'out' would be 0.
    second_half_norm = out[5:].norm()
    print(f"  Readout second half norm: {second_half_norm.item()}")
    assert (
        second_half_norm > 0.001
    ), "Readout output for second graph is zero! Indices likely overlapped."


def benchmark_transformation():
    """Benchmark the LineGraph transformation."""
    num_atoms = 100
    k = 20
    data = create_dummy_graph(num_atoms=num_atoms, num_neighbors=k)
    data.edge_attr = torch.randn(data.x.size(0), 1)

    transform = LineGraph()

    start = time.time()
    n_loops = 50
    for _ in range(n_loops):
        _ = transform(data.clone())
    end = time.time()

    avg_time = (end - start) / n_loops
    print(f"\nBenchmark LineGraph (atoms={num_atoms}, k={k}): {avg_time*1000:.4f} ms/iter")


if __name__ == "__main__":
    try:
        test_line_graph_transform_integrity()
        print("test_line_graph_transform_integrity: PASS")
    except Exception as e:
        print(f"test_line_graph_transform_integrity: FAIL - {e}")

    try:
        test_batching_indices()
        print("test_batching_indices: PASS")
    except Exception as e:
        print(f"test_batching_indices: FAIL - {e}")

    try:
        test_readout_on_batch()
        print("test_readout_on_batch: PASS")
    except Exception as e:
        print(f"test_readout_on_batch: FAIL - {e}")

    benchmark_transformation()
