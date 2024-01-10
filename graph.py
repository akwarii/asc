from datetime import timedelta
from functools import partial
import logging
from multiprocessing import Pool
from time import perf_counter
from typing import Optional

import numpy as np
import torch
from pymatgen.core.periodic_table import Element
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import Dataset
from tqdm import tqdm

logging.getLogger(__name__)

torch.set_default_dtype(torch.float32)


class Graph:
    """
    Graph object fro creation of atomic graphs with bond and node attributes from pymatgen structure
    """

    def __init__(
        self,
        neighbors=12,
        rcut=0,
        delta=1,
    ):
        self.neighbors = neighbors
        self.rcut = rcut
        self.delta = delta
        self.bond = []
        self.neighbor = []
        self.angle_cosines = []

    def set_features(self, structure):
        if self.rcut > 0:
            pass
        else:
            species = [site.specie.symbol for site in structure.sites]
            self.rcut = max([Element(elm).atomic_radius * 3 for elm in species])

        all_neighbors = structure.get_all_neighbors(self.rcut, include_index=True)

        len_neighbors = np.array([len(neighbor) for neighbor in all_neighbors])
        
        indexes = np.where((len_neighbors < self.neighbors))[0]

        for i in indexes:
            cut = self.rcut
            curr_N = len(all_neighbors[i])
            while curr_N < self.neighbors:
                cut += self.delta
                neighbor = structure.get_neighbors(structure[i], cut)
                curr_N = len(neighbor)
            all_neighbors[i] = neighbor

        all_neighbors = [
            sorted(neighbors, key=lambda x: x[1]) for neighbors in all_neighbors
        ]

        self.neighbor = torch.LongTensor(
            [
                list(map(lambda x: x[2], neighbors[: self.neighbors]))
                for neighbors in all_neighbors
            ]
        )
        self.bond = torch.Tensor(
            [
                list(map(lambda x: x[1], neighbors[: self.neighbors]))
                for neighbors in all_neighbors
            ]
        )

        cart_coords = torch.Tensor(
            np.array([structure[i].coords for i in range(len(structure))])
        )
        atom_neighbor_fea = torch.Tensor(
            np.array(
                [
                    list(map(lambda x: x[0].coords, neighbors[: self.neighbors]))
                    for neighbors in all_neighbors
                ]
            )
        )
        centre_coords = cart_coords.unsqueeze(1).expand(
            len(structure), self.neighbors, 3
        )
        dxyz = atom_neighbor_fea - centre_coords
        r = self.bond.unsqueeze(2)
        self.angle_cosines = torch.matmul(
            dxyz, torch.swapaxes(dxyz, 1, 2)
        ) / torch.matmul(r, torch.swapaxes(r, 1, 2))


def load_graphs_targets(data, neighbors=12, rcut=0, delta=1):
    """
    data should be in dict format
        structure:{pymatgen structure},
        property:{}
        formula: None or formula
    if not from database
    """
    structure = data["structure"]
    target = data["target"]

    graph = Graph(neighbors=neighbors, rcut=rcut, delta=delta)
    graph.set_features(structure)
    return (graph, target)


def process(func, tasks, mp_load: bool = False, n_proc: Optional[int] = None):
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


class CrystalGraphDataset(Dataset):
    def __init__(
            self,
            dataset,
            neighbors=12,
            rcut=0,
            delta=1,
            mp_load=False,
            mp_cpu_count=None,
            **kwargs,
        ):
            """
            Initialize the Graph object.

            Args:
                dataset: The dataset containing the graphs.
                neighbors: The number of neighbors to consider for each node (default: 12).
                rcut: The radius cutoff for the neighbor search (default: 0).
                delta: The delta parameter for the neighbor search (default: 1).
                mp_load: Whether to use multiprocessing for loading graphs (default: False).
                mp_cpu_count: The number of CPUs to use for multiprocessing (default: None).
                **kwargs: Additional keyword arguments.
            """
            logging.info(f"Loading {len(dataset)} graphs ...")
            print(f"Loading {len(dataset)} graphs ...")

            t1 = perf_counter()

            results = process(
                partial(load_graphs_targets, neighbors=neighbors, rcut=rcut, delta=delta),
                dataset,
                mp_load=mp_load,
                n_proc=mp_cpu_count,
            )

            self.graphs = [res[0] for res in results if res is not None]

            self.targets = [torch.LongTensor(res[1]) for res in results if res is not None]

            self.binarizer = LabelBinarizer()
            self.binarizer.fit(torch.cat(self.targets))
            self.num_classes = len(self.binarizer.classes_)

            t2 = perf_counter()
            logging.info(f"Graphs loaded in {timedelta(seconds=t2-t1)}s")
            print(f"Graphs loaded in {timedelta(seconds=t2-t1)}s")

            self.size = len(self.targets)

    def collate(self, datalist):
        bond_feature, neighbor_idx, angular_feature, crystal_idx, targets = (
            [],
            [],
            [],
            [],
            [],
        )

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

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        bond_feature = graph.bond
        neighbor_idx = graph.neighbor
        angular_feature = graph.angle_cosines
        target = self.targets[idx]

        return (bond_feature, neighbor_idx, angular_feature), target
