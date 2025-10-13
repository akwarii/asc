import time

import faiss
import faiss.contrib.torch_utils  # ignore
import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.build import bulk


def benchmark_knn_methods(sizes=None, k=20, repetitions=3):
    """Benchmark PyTorch vs FAISS KNN on different system sizes.

    Args:
        sizes: List of system sizes to test (atoms count)
        k: Number of neighbors to find
        repetitions: Number of runs for each size to average

    Returns:
        dict: Results containing execution times and memory usage
    """
    if sizes is None:
        # Test exponentially increasing sizes
        sizes = [
            100,
            200,
            500,
            1_000,
            2_000,
            5_000,
            10_000,
            20_000,
            50_000,
            100_000,
            200_000,
            500_000,
        ]

    results = {
        "sizes": sizes,
        "pytorch_time": [],
        "pytorch_memory": [],
        "faiss_time": [],
        "faiss_memory": [],
        "pytorch_oom": [],  # Track OOM failures
        "faiss_oom": [],  # Track OOM failures
    }

    # Track the maximum size each method can handle
    pytorch_max_size = float("inf")
    faiss_max_size = float("inf")

    for n_atoms in sizes:
        print(f"Benchmarking system with {n_atoms} atoms")

        # Skip if we already know this size will cause OOM
        if n_atoms > pytorch_max_size and n_atoms > faiss_max_size:
            print(f"  Skipping {n_atoms} atoms (both methods failed at smaller sizes)")
            results["pytorch_time"].append(None)
            results["pytorch_memory"].append(None)
            results["faiss_time"].append(None)
            results["faiss_memory"].append(None)
            results["pytorch_oom"].append(True)
            results["faiss_oom"].append(True)
            continue

        # Create a test system (cubic supercell of silicon)
        side_length = int(np.ceil(n_atoms ** (1 / 3)))
        atoms = bulk("Si", "diamond", a=5.43).repeat(side_length)

        # Truncate to exactly n_atoms, will not be perfect diamond structure but whatever
        if len(atoms) > n_atoms:
            del atoms[n_atoms:]

        # I don't think we could have less than n_atoms here but just in case
        assert len(atoms) == n_atoms, "Atom count mismatch"

        # Initialize PyTorch variables
        pytorch_times = []
        pytorch_memory = []
        pytorch_failed = False

        # Initialize FAISS variables
        faiss_times = []
        faiss_memory = []
        faiss_failed = False

        # Benchmark PyTorch method
        if n_atoms <= pytorch_max_size:
            for _ in range(repetitions):
                try:
                    torch.cuda.empty_cache()
                    start_memory = (
                        torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    )

                    start = time.time()
                    # Run the PyTorch KNN code
                    _ = run_pytorch_knn(atoms, k)
                    end = time.time()

                    peak_memory = (
                        torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
                    )
                    memory_used = peak_memory - start_memory

                    pytorch_times.append(end - start)
                    pytorch_memory.append(memory_used / 1e9)  # in GB

                # Tricky catching, I think it will do for Pytorch
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) or "cudaErrorMemoryAllocation" in str(e):
                        print(f"  PyTorch OOM at {n_atoms} atoms")
                        pytorch_failed = True
                        pytorch_max_size = n_atoms - 1  # Mark as failed for future iterations
                        break
                    else:
                        raise  # Re-raise if it's not an OOM error

            # Record results or mark as OOM
            if pytorch_failed:
                results["pytorch_time"].append(None)
                results["pytorch_memory"].append(None)
                results["pytorch_oom"].append(True)
            else:
                results["pytorch_time"].append(np.mean(pytorch_times))
                results["pytorch_memory"].append(np.mean(pytorch_memory))
                results["pytorch_oom"].append(False)
        else:
            # To win time: skip benchmark for known OOM sizes
            results["pytorch_time"].append(None)
            results["pytorch_memory"].append(None)
            results["pytorch_oom"].append(True)
            print(f"  Skipping PyTorch benchmark for {n_atoms} atoms (OOM expected)")

        # Benchmark FAISS method -- mostly same as above
        if n_atoms <= faiss_max_size:
            for _ in range(repetitions):
                try:
                    torch.cuda.empty_cache()
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                    start_memory = (
                        torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    )

                    start = time.time()
                    # Run the FAISS KNN code
                    _ = run_faiss_knn(atoms, k)
                    end = time.time()

                    peak_memory = (
                        torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
                    )
                    memory_used = peak_memory - start_memory
                    # TODO: Check if FAISS allocates memory outside of PyTorch

                    faiss_times.append(end - start)
                    faiss_memory.append(memory_used / 1e9)  # in GB
                # Tricky catching, not sure if this will catch all FAISS OOM errors
                except RuntimeError as e:
                    # ? DB : Never met faiss OOM so unsure what to catch
                    if (
                        "CUDA out of memory" in str(e)
                        or "cudaErrorMemoryAllocation" in str(e)
                        or "No memory" in str(e)
                    ):
                        print(f"  FAISS OOM at {n_atoms} atoms")
                        faiss_failed = True
                        faiss_max_size = n_atoms - 1  # Mark as failed for future iterations
                        break
                    else:
                        raise  # Re-raise if it's not an OOM error

            # Record results or mark as OOM
            if faiss_failed:
                results["faiss_time"].append(None)
                results["faiss_memory"].append(None)
                results["faiss_oom"].append(True)
            else:
                results["faiss_time"].append(np.mean(faiss_times))
                results["faiss_memory"].append(np.mean(faiss_memory))
                results["faiss_oom"].append(False)
        else:
            # To win time: skip benchmark for known OOM sizes
            results["faiss_time"].append(None)
            results["faiss_memory"].append(None)
            results["faiss_oom"].append(True)
            print(f"  Skipping FAISS benchmark for {n_atoms} atoms (OOM expected)")

    # Find crossover point for time
    crossover_time = find_crossover_point(sizes, results["pytorch_time"], results["faiss_time"])

    # Find crossover point for memory
    crossover_memory = find_crossover_point(
        sizes, results["pytorch_memory"], results["faiss_memory"]
    )

    print(f"Time crossover point: ~{crossover_time} atoms")
    print(f"Memory crossover point: ~{crossover_memory} atoms")

    # Print OOM thresholds
    print(f"PyTorch max size before OOM: {pytorch_max_size} atoms")
    print(f"FAISS max size before OOM: {faiss_max_size} atoms")

    # Plot results
    plot_benchmark_results(results, crossover_time, crossover_memory)

    return results, crossover_time, crossover_memory


def run_pytorch_knn(atoms, k):
    """PyTorch KNN calculation directly copied from the code

    Args:
        atoms: ASE Atoms object
        k: Number of neighbors to find

    Returns:
        neighbors_idx: Indices of the k nearest neighbors for each atom
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract coordinates and lattice
    cart_coords = torch.as_tensor(atoms.positions, dtype=torch.float32, device=device)
    lat = torch.as_tensor(atoms.cell.array, dtype=torch.float32, device=device)

    # Build periodic images
    shifts = torch.tensor(
        [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=torch.int32,
        device=device,
    )
    shifts_cart = shifts.to(torch.float32) @ lat

    imgs_cart = cart_coords.unsqueeze(0) + shifts_cart.unsqueeze(1)
    pts = imgs_cart.reshape(-1, 3).contiguous()

    # Calculate KNN using pure PyTorch
    a_norm = torch.sum(cart_coords**2, dim=1, keepdim=True)
    b_norm = torch.sum(pts**2, dim=1).view(1, -1)
    squared_dists = a_norm + b_norm - 2 * torch.mm(cart_coords, pts.t())

    distances, neighbors_idx = torch.topk(
        squared_dists, k=k + 1, dim=1, largest=False, sorted=False
    )

    distances = torch.sqrt(distances)  # Not really needed

    return neighbors_idx  # Most likely not needed


def run_faiss_knn(atoms, k):
    """FAISS KNN calculation directly copied from the code

    Args:
        atoms: ASE Atoms object
        k: Number of neighbors to find

    Returns:
        neighbors_idx: Indices of the k nearest neighbors for each atom
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract coordinates and lattice
    cart_coords = torch.as_tensor(atoms.positions, dtype=torch.float32, device=device)
    lat = torch.as_tensor(atoms.cell.array, dtype=torch.float32, device=device)

    # Build periodic images
    shifts = torch.tensor(
        [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=torch.int32,
        device=device,
    )
    shifts_cart = shifts.to(torch.float32) @ lat

    imgs_cart = cart_coords.unsqueeze(0) + shifts_cart.unsqueeze(1)
    pts = imgs_cart.reshape(-1, 3).contiguous()

    # Calculate KNN using FAISS
    is_faiss_gpu = hasattr(faiss, "StandardGpuResources")
    if is_faiss_gpu and torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        squared_dist, neighbors_idx = faiss.knn_gpu(res, cart_coords.contiguous(), pts, k + 1)
    else:
        squared_dist, neighbors_idx = faiss.knn(cart_coords.contiguous(), pts, k + 1)

    return neighbors_idx  # Most likely not needed


def find_crossover_point(sizes, metric1, metric2):
    """Tries to linearly estimate the approximate crossover point between two metrics

    Args:
        sizes: List of system sizes (atoms count)
        metric1: List of metric values for method 1 (e.g., PyTorch)
        metric2: List of metric values for method 2 (e.g., FAISS)
    Returns:
        int or float: Estimated crossover size, or float('inf') if no crossover found
    """
    # OOM values are None, which we don't need -> filter them out
    valid_indices = [
        i for i in range(len(sizes)) if metric1[i] is not None and metric2[i] is not None
    ]

    if not valid_indices:
        print("No valid data points to determine crossover")
        return float("inf")  # No valid comparison points

    # Filtered lists -- not sure if it can be done more elegantly
    valid_sizes = [sizes[i] for i in valid_indices]
    valid_metric1 = [metric1[i] for i in valid_indices]  # Would be PyTorch, but can be either
    valid_metric2 = [metric2[i] for i in valid_indices]  # Is the other one, likely FAISS

    for i in range(len(valid_sizes) - 1):
        # Find the interval where the crossover happens in the lists
        if valid_metric1[i] <= valid_metric2[i] and valid_metric1[i + 1] > valid_metric2[i + 1]:
            # Points to consider for linear interpolation
            x1, y1 = valid_sizes[i], valid_metric1[i] - valid_metric2[i]  # y1 < 0
            x2, y2 = valid_sizes[i + 1], valid_metric1[i + 1] - valid_metric2[i + 1]  # y2 > 0
            # Find where y = 0 using linear interpolation
            crossover = x1 - y1 * (x2 - x1) / (y2 - y1)
            return int(crossover)
        # Not likely to happen in our case, would mean Pytorch being slower than FAISS
        # for small sizes and faster for large ones --> just in case arguments are swapped
        elif valid_metric1[i] > valid_metric2[i] and valid_metric1[i + 1] <= valid_metric2[i + 1]:
            x1, y1 = valid_sizes[i], valid_metric1[i] - valid_metric2[i]
            x2, y2 = valid_sizes[i + 1], valid_metric1[i + 1] - valid_metric2[i + 1]
            crossover = x1 - y1 * (x2 - x1) / (y2 - y1)
            return int(crossover)

    # No crossover found
    if valid_metric1[-1] < valid_metric2[-1]:
        return float("inf")  # PyTorch always better
    else:
        return 0  # FAISS always better


def plot_benchmark_results(results, crossover_time, crossover_memory):
    """Plot benchmark results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sizes = np.array(results["sizes"])

    # Create masks for valid data points
    pytorch_mask = np.array([x is not None for x in results["pytorch_time"]])
    faiss_mask = np.array([x is not None for x in results["faiss_time"]])

    # Time plot
    if any(pytorch_mask):
        pytorch_times = np.array([t if t is not None else np.nan for t in results["pytorch_time"]])
        ax1.plot(
            sizes[pytorch_mask], pytorch_times[pytorch_mask], "o-", label="PyTorch", color="orange"
        )

        # Mark OOM points if any
        if any(results["pytorch_oom"]):
            oom_idx = np.argmax(results["pytorch_oom"])
            # ? From `np.argmax` docstring -> In case of multiple occurrences
            # ? of the maximum values, the indices corresponding to the first
            # ? occurrence are returned.
            # ? It's a lot faster than `np.argwhere` as we only need one index.
            if oom_idx > 0 and oom_idx < len(sizes):
                ax1.axvline(x=sizes[oom_idx], color="orangered", linestyle="--", alpha=0.5)
                ax1.text(
                    sizes[oom_idx] * 0.9,
                    pytorch_times[oom_idx - 1],
                    f"PyTorch OOM : {sizes[oom_idx]} atoms",
                    rotation=0,  # 90° is not readable
                    color="orangered",
                    alpha=0.9,
                    ha="right",
                )

    if any(faiss_mask):
        faiss_times = np.array([t if t is not None else np.nan for t in results["faiss_time"]])
        ax1.plot(sizes[faiss_mask], faiss_times[faiss_mask], "s-", label="FAISS", color="teal")

        # Mark OOM points if any
        if any(results["faiss_oom"]):
            oom_idx = np.argmax(results["faiss_oom"])  # same as above
            if oom_idx > 0 and oom_idx < len(sizes):
                ax1.axvline(x=sizes[oom_idx], color="darkslategrey", linestyle="--", alpha=0.5)
                ax1.text(
                    sizes[oom_idx] * 0.9,
                    faiss_times[oom_idx - 1],
                    f"FAISS OOM : {sizes[oom_idx]} atoms",
                    rotation=0,
                    color="darkslategrey",
                    alpha=0.9,
                    ha="right",
                )

    # Add crossover line if applicable
    if crossover_time != float("inf") and crossover_time != 0:
        ax1.axvline(x=crossover_time, color="dimgray", linestyle="--", alpha=0.5)
        ax1.text(
            crossover_time * 0.9,
            ax1.get_ylim()[1] * 0.995,
            f"Time Crossover ~ {crossover_time} atoms",
            rotation=0,
            color="dimgray",
            alpha=0.9,
            ha="right",
            va="top",
        )

    ax1.set_xlabel("Number of atoms")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("KNN Calculation Time")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    # ax1.grid(which="both", linestyle="--", alpha=0.7)  # Disabled for clarity
    ax1.legend()

    # Memory plot
    if any(pytorch_mask):
        pytorch_memory = np.array(
            [m if m is not None else np.nan for m in results["pytorch_memory"]]
        )
        ax2.plot(
            sizes[pytorch_mask],
            pytorch_memory[pytorch_mask],
            "o-",
            label="PyTorch",
            color="orange",
        )
        # Add the maximum memory line
        max_memory = max(pytorch_memory[pytorch_mask])  # Regardless of OOM
        ax2.axhline(y=max_memory, color="orangered", linestyle="--", alpha=0.5)
        ax2.text(
            sizes[pytorch_mask][-1],
            max_memory,
            f"PyTorch Max Memory: {max_memory:.2f} GB",
            color="orangered",
            alpha=0.9,
            ha="right",
            va="bottom",
        )

    if any(faiss_mask):
        faiss_memory = np.array([m if m is not None else np.nan for m in results["faiss_memory"]])
        ax2.plot(sizes[faiss_mask], faiss_memory[faiss_mask], "s-", label="FAISS", color="teal")
        # Add the maximum memory line
        max_memory = max(faiss_memory[faiss_mask])  # Regardless of OOM
        ax2.axhline(y=max_memory, color="darkslategrey", linestyle="--", alpha=0.5)
        ax2.text(
            sizes[faiss_mask][-1],
            max_memory,
            f"FAISS Max Memory: {max_memory:.2f} GB",
            color="darkslategrey",
            alpha=0.9,
            ha="right",
            va="bottom",
        )

    # Add crossover line if applicable
    if crossover_memory != float("inf") and crossover_memory != 0:
        ax2.axvline(x=crossover_memory, color="dimgray", linestyle="--", alpha=0.5)
        ax2.text(
            crossover_memory * 0.9,
            ax2.get_ylim()[1] * 0.995,
            f"Memory Crossover ~ {crossover_memory} atoms",
            rotation=0,
            color="dimgray",
            alpha=0.9,
            ha="right",
            va="top",
        )

    ax2.set_xlabel("Number of atoms")
    ax2.set_ylabel("Memory Usage (GB)")
    ax2.set_title("KNN Memory Usage")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    # ax2.grid(which="both", linestyle="--", alpha=0.7)  # Disabled for clarity
    ax2.legend()

    plt.tight_layout()
    plt.savefig("knn_benchmark_results.png")
    #plt.show()


if __name__ == "__main__":
    # Start with smaller sizes first to avoid immediate OOM errors
    sizes = [
        1,
        2,
        5,
        10,
        25,
        50,
        75,
        100,
        200,
        500,
        750,
        1_000,
        2_500,
        5_000,   # Close to the local GPU crossover region (NVIDIA RTX 4000 Ada Generation)
        6_000,   # Close to the local GPU crossover region (NVIDIA RTX 4000 Ada Generation)
        6_125,   # Close to the local GPU crossover region (NVIDIA RTX 4000 Ada Generation)
        6_250,   # Close to the local GPU crossover region (NVIDIA RTX 4000 Ada Generation)
        6_500,   # Close to the local GPU crossover region (NVIDIA RTX 4000 Ada Generation)
        6_750,   # Close to the local GPU crossover region (NVIDIA RTX 4000 Ada Generation)
        7_000,   # Close to the local GPU crossover region (NVIDIA RTX 4000 Ada Generation)
        7_500,   # Close to the local GPU crossover region (NVIDIA RTX 4000 Ada Generation)
        10_000,
        10_100,  # Close to the IN2P3 GPU crossover region (NVIDIA V100)
        10_150,  # Close to the IN2P3 GPU crossover region (NVIDIA V100)
        10_175,  # Close to the IN2P3 GPU crossover region (NVIDIA V100)
        10_200,  # Close to the IN2P3 GPU crossover region (NVIDIA V100)
        10_500,  # Close to the IN2P3 GPU crossover region (NVIDIA V100)
        11_000,
        12_500,
        15_000,
        17_250,  # Close to the IN2P3 GPU crossover region (NVIDIA H100)
        17_400,  # Close to the IN2P3 GPU crossover region (NVIDIA H100)
        17_450,  # Close to the IN2P3 GPU crossover region (NVIDIA H100)
        17_475,  # Close to the IN2P3 GPU crossover region (NVIDIA H100)
        17_500,  # Close to the IN2P3 GPU crossover region (NVIDIA H100)
        20_000,
    ]
    results, time_crossover, memory_crossover = benchmark_knn_methods(sizes=sizes, k=20)

    # Determine the recommended thresholds
    if time_crossover != float("inf") and memory_crossover != float("inf"):
        print("Recommended thresholds for switching to FAISS:")
        print(f"  * For memory = {memory_crossover} atoms")
        print(f"  * For time   = {time_crossover} atoms")
    else:
        # Handle case where no crossover was found
        if time_crossover == float("inf") and memory_crossover == float("inf"):
            print("PyTorch was better for all tested sizes")
        elif time_crossover == 0 and memory_crossover == 0:
            print("FAISS was better for all tested sizes")
        else:
            print(f"Mixed results: Time crossover   = {time_crossover}")
            print(f"               Memory crossover = {memory_crossover}")
