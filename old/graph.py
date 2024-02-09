from functools import partial
from typing import Optional

import numpy as np
import torch
from pymatgen.core import IStructure
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import Dataset

from src.utils.bonds import compute_bond_angle_cosines
from src.utils.data import load_graphs_and_targets, process
from src.utils.neighbors import find_knn_in_shell

torch.set_default_dtype(torch.float32)


class Graph:
    """Graph object for creation of atomic graphs with bond and node attributes from pymatgen
    structure."""

    def __init__(
        self,
        neighbors: int = 12,
        rcut: float = 0,
        delta: float = 1,
    ) -> None:
        self.n_neighbors = neighbors
        self.rcut = rcut
        self.delta = delta

    def set_features(self, structure: IStructure) -> None:
        all_neighbors_sorted = find_knn_in_shell(
            structure, self.rcut, self.n_neighbors, self.delta
        )

        # Graph features (nodes: atoms, edges: bonds)
        # TODO add one-hot encoding of atomic numbers
        self.edge_features = torch.from_numpy(
            np.array(
                [[x.nn_distance for x in neighbors] for neighbors in all_neighbors_sorted],
                dtype=np.float32,
            )
        )
        self.neighbor_list = torch.from_numpy(
            np.array(
                [[x.index for x in neighbors] for neighbors in all_neighbors_sorted],
                dtype=np.int32,
            )
        )

        # Line graph features (nodes: bonds, edges: angles)
        self.line_graph_features = compute_bond_angle_cosines(
            structure, all_neighbors_sorted, self.edge_features
        )


class CrystalGraphDataset(Dataset):
    """Dataset class for crystal graph data.

    Args:
        dataset (list[dict[str, np.ndarray | IStructure]]): List of dictionaries containing the dataset.
        neighbors (int, optional): Number of neighbors to consider. Defaults to 12.
        rcut (float, optional): Cutoff radius. Defaults to 0.
        delta (float, optional): Delta value. Defaults to 1.
        mp_load (bool, optional): Whether to use multiprocessing for loading graphs. Defaults to False.
        mp_cpu_count (Optional[int], optional): Number of CPUs to use for multiprocessing. Defaults to None.

    Attributes:
        graphs (list): List of loaded graphs.
        targets (list): List of targets.
        num_classes (int): Number of classes.
        size (int): Size of the dataset.

    Methods:
        collate: Collates the data.
        __getitem__: Retrieves an item from the dataset.
    """

    def __init__(
        self,
        dataset: list[dict[str, np.ndarray | IStructure]],
        neighbors: int = 12,
        rcut: float = 0,
        delta: float = 1,
        mp_load: bool = False,
        mp_cpu_count: int | None = None,
    ) -> None:
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")

        results = process(
            partial(load_graphs_and_targets, neighbors=neighbors, rcut=rcut, delta=delta),
            dataset,
            mp_load=mp_load,
            n_proc=mp_cpu_count,
        )

        self.graphs: list[Graph] = [res[0] for res in results if res is not None]
        self.targets = [torch.ShortTensor(res[1]) for res in results if res is not None]

        # TODO: pretty sure this is useless as it is, but I'll leave it for now
        binarizer = LabelBinarizer()
        binarizer.fit(torch.cat(self.targets))
        self.num_classes = len(binarizer.classes_)

    @property
    def size(self) -> int:
        return len(self.graphs)

    def collate(
        self, datalist
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bond_feature, neighbor_idx, angular_feature, crystal_idx, targets = [], [], [], [], []
        index = 0

        for (bond_fea, idx, angular_fea), targ in datalist:
            natoms = bond_fea.shape[0]

            bond_feature.append(bond_fea)
            angular_feature.append(angular_fea)
            neighbor_idx.append(idx + index)
            crystal_idx.append([index, index + natoms])
            targets.append(targ)
            index += natoms

        return (
            torch.cat(bond_feature, dim=0),
            torch.cat(angular_feature, dim=0),
            torch.cat(neighbor_idx, dim=0),
            torch.LongTensor(crystal_idx),
            torch.cat(targets, dim=0),
        )

    def __getitem__(self, idx: int):
        graph = self.graphs[idx]
        bond_feature = graph.edge_features
        neighbor_idx = graph.neighbor_list
        angular_feature = graph.line_graph_features
        target = self.targets[idx]

        return (bond_feature, neighbor_idx, angular_feature), target


if __name__ == "__main__":
    from utils import load_dataset, load_settings

    dataset_path = "example/train"
    settings = load_settings("config/test_config.yaml")
    graphs = load_dataset(dataset_path, settings)
    print(graphs.graphs[0].edge_features.shape)
    print(graphs.graphs[0].neighbor_list.shape)
    print(graphs.graphs[0].line_graph_features.shape)
