from functools import partial
from typing import Any, Callable, Optional

import numpy as np
import torch
from mp_api.client import MPRester
from pymatgen.core import IStructure
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import Dataset

from src.utils.bonds import compute_bond_angle_cosines
from src.utils.data import load_graphs_and_targets, process
from src.utils.neighbors import find_knn_in_shell


class Graph:
    """
    Graph object for creation of atomic graphs with bond and node attributes from pymatgen structure
    """

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
        #TODO add one-hot encoding of atomic numbers
        self.edge_features = torch.from_numpy(
            np.array(
                [
                    [x.nn_distance for x in neighbors]
                    for neighbors in all_neighbors_sorted
                ],
                dtype=np.float32,
            )
        )
        self.neighbor_list = torch.from_numpy(
            np.array(
                [
                    [x.index for x in neighbors]
                    for neighbors in all_neighbors_sorted
                ],
                dtype=np.int32,
            )
        )

        # Line graph features (nodes: bonds, edges: angles)
        self.line_graph_features = compute_bond_angle_cosines(
            structure, all_neighbors_sorted, self.edge_features
        )


class CrystalGraphDataset(Dataset):

    mirrors = [
        "https://aflow.org/API/aflux/?",
        MPRester
    ]

    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()
        
    def _load_data(self):
        results = process(
            partial(load_graphs_and_targets, neighbors=neighbors, rcut=rcut, delta=delta),
            dataset,
            mp_load=mp_load,
            n_proc=mp_cpu_count,
        )

        self.data: list[Graph] = [res[0] for res in results if res is not None]
        self.targets = [torch.ShortTensor(res[1]) for res in results if res is not None]

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int):
        graph = self.data[idx]
        bond_feature = graph.edge_features
        neighbor_idx = graph.neighbor_list
        angular_feature = graph.line_graph_features
        target = self.targets[idx]

        return (bond_feature, neighbor_idx, angular_feature), target
    
    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        graph, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            graph = self.transform(graph)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return graph, target

    def collate(self, datalist) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

