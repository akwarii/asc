from multiprocessing import Pool
from typing import Any, Callable, Iterable, Optional

import torch
from tqdm import tqdm

from src.processing.graph import Graph


def load_graphs_and_targets(
    data: dict[str, Any],
    neighbors: int = 12,
    rcut: float = 0,
    delta: float = 1
) -> tuple[Graph, torch.Tensor]:
    """
    Create graphs from Pymatgen structures and return both graphs and targets.

    Args:
        data (dict[str, Any]): Dictionary containing the structure and target.
        neighbors (int, optional): Nnumber of neighbors to consider. Defaults to 12.
        rcut (float, optional): Cutoff radius. Defaults to 0.
        delta (float, optional): Delta parameter used to increase rcut stepwise. Defaults to 1.

    Returns:
        tuple[Graph, torch.Tensor]: A tuple containing the loaded graph and target.
    """
    structure = data["structure"]
    target = data["target"]

    graph = Graph(neighbors=neighbors, rcut=rcut, delta=delta)
    graph.set_features(structure)
    return graph, target


def process(
    func: Callable, tasks: Iterable, mp_load: bool = False, n_proc: Optional[int] = None
) -> list:
    """
    Process the given tasks using the provided function.

    Args:
        func (Callable): The function to be applied to each task.
        tasks (Iterable): The tasks to be processed.
        mp_load (bool, optional): Whether to use multiprocessing. Defaults to False.
        n_proc (int, optional): The number of processes to use if multiprocessing is enabled. Defaults to None.

    Returns:
        list: The results of applying the function to each task.
    """
    if mp_load:
        with Pool(n_proc) as mp_pool:
            results = []
            chunks = [tasks[i : i + n_proc] for i in range(0, len(tasks), n_proc)]
            for chunk in chunks:
                r = mp_pool.map_async(func, chunk, callback=results.append)
                r.wait()
            mp_pool.close()
            mp_pool.join()
        return results[0]
    else:
        return [func(task) for task in tqdm(tasks, desc="Building graphs")]