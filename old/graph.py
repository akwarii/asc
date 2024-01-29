import logging
from functools import partial
from multiprocessing import Pool
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch
from pymatgen.core import IStructure
from pymatgen.core.periodic_table import Element
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import Dataset
from tqdm import tqdm

logging.getLogger(__name__)

torch.set_default_dtype(torch.float32)

#TODO use Deep Graph Library (DGL) to create the graphs may be faster and cleaner
#TODO we may be intered in the sparse matrix representation of the graph
class Graph:
    """
    Graph object for creation of atomic graphs with bond and node attributes from pymatgen structure
    """

    def __init__(
        self,
        neighbors: int = 12,
        rcut: float = 0,
        delta: float = 1,
    ):
        self.n_neighbors = neighbors
        self.rcut = rcut
        self.delta = delta
        self.bond = []
        self.neighbor = []
        self.angle_cosines = []

    def set_features(self, structure: IStructure):
        if self.rcut <= 0:
            species = [site.specie.symbol for site in structure.sites]
            self.rcut = max([Element(elm).atomic_radius * 3 for elm in species])

        all_neighbors = structure.get_all_neighbors(self.rcut, include_index=True)
        len_neighbors = np.array([len(neighbor) for neighbor in all_neighbors])
        missing_neighbors_idx = np.where((len_neighbors < self.n_neighbors))[0]

        for i in missing_neighbors_idx:
            rcut = self.rcut
            n_neighbors = len(all_neighbors[i])
            while n_neighbors < self.n_neighbors:
                rcut += self.delta
                neighbor = structure.get_neighbors(structure[i], rcut)
                n_neighbors = len(neighbor)
            all_neighbors[i] = neighbor

        all_neighbors_sorted = [
            sorted(neighbors, key=lambda x: x[1])[:self.n_neighbors] for neighbors in all_neighbors
        ]

        atom_neighbor_fea = torch.from_numpy(np.array([[x[0].coords for x in neighbors] for neighbors in all_neighbors_sorted], dtype=np.float32))
        self.bond = torch.from_numpy(np.array([[x[1] for x in neighbors] for neighbors in all_neighbors_sorted], dtype=np.float32)) # bond length
        self.neighbor = torch.from_numpy(np.array([[x[2] for x in neighbors] for neighbors in all_neighbors_sorted], dtype=np.int32)) # neighbor index (ie node labels)

        cartesian_coords = torch.from_numpy(structure.cart_coords).float()
        centre_coords = cartesian_coords.unsqueeze(1).expand(
            len(structure), self.n_neighbors, 3
        )
        dxyz = atom_neighbor_fea - centre_coords
        r = self.bond.unsqueeze(2)
        self.angle_cosines = torch.matmul(
            dxyz, torch.swapaxes(dxyz, 1, 2)
        ) / torch.matmul(r, torch.swapaxes(r, 1, 2))  # cosine rule
    

def load_graphs_targets(data: dict[str, Any], neighbors: int = 12, rcut: float = 0, delta: float = 1) -> tuple[Graph, torch.Tensor]:
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


def process(func: Callable, tasks: Iterable, mp_load: bool = False, n_proc: Optional[int] = None) -> list:
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
    """
    Dataset class for crystal graph data.

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
        mp_cpu_count: Optional[int] = None,
    ) -> None:
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")
        
        results = process(
            partial(load_graphs_targets, neighbors=neighbors, rcut=rcut, delta=delta),
            dataset,
            mp_load=mp_load,
            n_proc=mp_cpu_count,
        )

        self.graphs: list[Graph] = [res[0] for res in results if res is not None]

        self.targets = [torch.ShortTensor(res[1]) for res in results if res is not None]

        #TODO: look into this, I think the one-hot encoding is never used but maybe it behaves differently
        # and assigns the values directly to the tensor
        binarizer = LabelBinarizer()
        binarizer.fit(torch.cat(self.targets))
        self.num_classes = len(binarizer.classes_)

    @property
    def size(self) -> int:
        return len(self.graphs)
    
    def collate(self, datalist) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def __getitem__(self, idx: int):
        graph = self.graphs[idx]
        bond_feature = graph.bond
        neighbor_idx = graph.neighbor
        angular_feature = graph.angle_cosines
        target = self.targets[idx]

        return (bond_feature, neighbor_idx, angular_feature), target


if __name__ == "__main__":
    from utils import load_dataset, load_settings
    
    dataset_path = "example/train"
    settings = load_settings("config/test_config.yaml")
    graphs = load_dataset(dataset_path, settings)
    print(graphs.graphs[0].bond.shape)
    print(graphs.graphs[0].neighbor.shape)
    print(graphs.graphs[0].angle_cosines.shape)
    print(graphs.targets[0].shape)
    print(graphs.graphs[0].bond)