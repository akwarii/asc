import time
from pathlib import Path

import torch
from src.graph import KNNGraph
from src.models.cegann_v2 import CEGANNv2
from src.transforms.line_graph import LineGraph
from torch_geometric.data import Data


def load_test_structures(data_file: Path, num_samples: int = 10) -> list[Path]:
    """Load test structures from the data directory."""
    import pandas as pd

    df = pd.read_csv(data_file)
    structure_files = df["Structure"].tolist()
    return structure_files[:num_samples]


def convert_structures_to_graphs(structure_files: list[Path], k: int = 20) -> list[Data]:
    """Convert structures to PyG Data objects using KNNGraph."""
    print(f"\nConverting {len(structure_files)} structures to graphs with k={k}...")

    knn_graph = KNNGraph(k=k)
    graphs = []

    for structure_file in structure_files:
        try:
            graph = knn_graph.convert(structure_file)
            graphs.append(graph)
        except Exception as e:
            print(f"Warning: Failed to convert {structure_file.name}: {e}")
            continue

    print(f"Successfully converted {len(graphs)} structures to graphs.")
    return graphs


def test_line_graph_transform(
    graphs: list[Data],
    *,
    delay_angle_computation: bool,
) -> tuple[list[Data], float]:
    """Apply LineGraph transform and measure time."""
    transform = LineGraph(delay_angle_computation=delay_angle_computation)

    transformed_graphs = []
    start_time = time.perf_counter()

    for graph in graphs:
        # Clone to avoid modifying original
        graph_copy = graph.clone()
        transformed = transform(graph_copy)
        transformed_graphs.append(transformed)

    elapsed_time = time.perf_counter() - start_time

    return transformed_graphs, elapsed_time


def compare_results(graphs_immediate: list[Data], graphs_delayed: list[Data]) -> None:
    """Compare results from immediate vs delayed angle computation."""
    print("\nComparing immediate vs delayed angle computation:")

    num_graphs = min(len(graphs_immediate), len(graphs_delayed))

    all_match = True
    for i in range(num_graphs):
        g_imm = graphs_immediate[i]
        g_del = graphs_delayed[i]

        # Check structure
        assert g_imm.num_nodes == g_del.num_nodes, f"Graph {i}: num_nodes mismatch"
        assert g_imm.edge_index.shape == g_del.edge_index.shape, (
            f"Graph {i}: edge_index shape mismatch"
        )
        assert torch.allclose(g_imm.edge_index, g_del.edge_index), (
            f"Graph {i}: edge_index mismatch"
        )

        # Check node features (distances)
        assert torch.allclose(g_imm.x, g_del.x, atol=1e-6), f"Graph {i}: node features mismatch"

        # Check edge attributes (angles)
        if g_imm.edge_attr is not None and g_del.edge_attr is not None:
            if not torch.allclose(g_imm.edge_attr, g_del.edge_attr, atol=1e-6):
                print(f"  Graph {i}: edge_attr differs slightly")
                max_diff = (g_imm.edge_attr - g_del.edge_attr).abs().max().item()
                print(f"    Max difference: {max_diff:.2e}")
                all_match = False

    if all_match:
        print("  ✓ All graphs match exactly!")
    else:
        print("  ⚠ Some minor differences detected (likely numerical precision)")


def benchmark_approaches(
    graphs: list[Data],
    runs: int = 5,
) -> None:
    """Benchmark both approaches multiple times."""
    print(f"\nBenchmarking with {runs} runs on {len(graphs)} graphs...")

    immediate_times = []
    delayed_transform_times = []
    delayed_compute_times = []

    for run in range(runs):
        # Test immediate computation
        _, imm_time = test_line_graph_transform(graphs, delay_angle_computation=False)
        immediate_times.append(imm_time)

        # Test delayed computation
        delayed_graphs, del_trans_time = test_line_graph_transform(
            graphs, delay_angle_computation=True
        )
        del_comp_time = compute_delayed_angles(delayed_graphs)

        delayed_transform_times.append(del_trans_time)
        delayed_compute_times.append(del_comp_time)

    # Compute statistics
    imm_mean = torch.tensor(immediate_times).mean().item()
    imm_std = torch.tensor(immediate_times).std().item()

    del_trans_mean = torch.tensor(delayed_transform_times).mean().item()
    del_trans_std = torch.tensor(delayed_transform_times).std().item()

    del_comp_mean = torch.tensor(delayed_compute_times).mean().item()
    del_comp_std = torch.tensor(delayed_compute_times).std().item()

    del_total_mean = del_trans_mean + del_comp_mean
    del_total_std = (
        torch.tensor([t + c for t, c in zip(delayed_transform_times, delayed_compute_times)])
        .std()
        .item()
    )

    print("\nBenchmark Results:")
    print(f"  Immediate computation:     {imm_mean * 1000:.2f} ± {imm_std * 1000:.2f} ms")
    print(
        f"  Delayed - transform only:  {del_trans_mean * 1000:.2f} ± {del_trans_std * 1000:.2f} ms"
    )
    print(
        f"  Delayed - angle compute:   {del_comp_mean * 1000:.2f} ± {del_comp_std * 1000:.2f} ms"
    )
    print(
        f"  Delayed - total:           {del_total_mean * 1000:.2f} ± {del_total_std * 1000:.2f} ms"
    )

    speedup = (imm_mean / del_total_mean - 1) * 100
    print(f"\n  Transform-only speedup: {(1 - del_trans_mean / imm_mean) * 100:.1f}%")
    print(f"  Overall speedup: {speedup:+.1f}%")


def main() -> None:
    torch.manual_seed(42)

    # Configuration
    data_file = Path("data/csg/raw/CSG_tiny.csv")
    num_samples = 1_000
    k_neighbors = 20
    benchmark_runs = 5
    num_classes = 10

    print("=" * 60)
    print("Testing Line Graph Transform with Online Angle Computation")
    print("=" * 60)

    # Load structures
    print(f"\nLoading structures from {data_file}...")
    structure_files = load_test_structures(data_file, num_samples=num_samples)
    print(f"Loaded {len(structure_files)} structure files.")

    # Convert to graphs
    graphs = convert_structures_to_graphs(structure_files, k=k_neighbors)

    if not graphs:
        print("No graphs to test. Exiting.")
        return

    # Test both approaches once
    print("\n" + "=" * 60)
    print("Testing graph construction with immediate angle computation...")
    graphs_immediate, time_immediate = test_line_graph_transform(
        graphs, delay_angle_computation=False
    )
    print(f"Time: {time_immediate * 1000:.2f} ms")

    print("\n" + "=" * 60)
    print("Testing graph construction with delayed angle computation...")
    graphs_delayed, time_delayed = test_line_graph_transform(graphs, delay_angle_computation=True)
    print(f"Transform time: {time_delayed * 1000:.2f} ms")

    # Initialize cegann_v2 model
    model = CEGANNv2(
        num_classes=num_classes,
        emb_num_radial=16,
        emb_num_angular=16,
        emb_node_out_channels=16,
        emb_edge_out_channels=16,
        emb_hidden_channels=16,
    )

    # Use GPU if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = model.to(device)

    # Forward pass with precomputed angles
    print("\n" + "=" * 60)
    print("Performing forward pass with immediate angle computation...")
    start_time = time.perf_counter()
    out_immediate = torch.empty(num_samples, num_classes)
    for i, g in enumerate(graphs_immediate):
        # out_immediate[i] = model(g.to(device))
        _ = model(g.to(device))
    elapsed_time = time.perf_counter() - start_time
    print(f"Output shape (immediate): {out_immediate.shape}")
    print(f"Forward pass time (immediate): {elapsed_time * 1000:.2f} ms")

    # Forward pass with delayed angles
    print("\n" + "=" * 60)
    print("Performing forward pass with delayed angle computation...")
    start_time = time.perf_counter()
    out_delayed = torch.empty(num_samples, num_classes)
    for i, g in enumerate(graphs_delayed):
        # out_delayed[i] = model(g.to(device))
        _ = model(g.to(device))
    elapsed_time = time.perf_counter() - start_time
    print(f"Output shape (delayed): {out_delayed.shape}")
    print(f"Forward pass time (delayed): {elapsed_time * 1000:.2f} ms")

    # Compare results
    compare_results(graphs_immediate, graphs_delayed)

    # Detailed benchmark
    print("\n" + "=" * 60)
    benchmark_approaches(graphs, runs=benchmark_runs)

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
