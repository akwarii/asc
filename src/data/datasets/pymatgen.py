from collections.abc import Callable
from pathlib import Path
from typing import Any

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from torch_geometric.data import Data

from src.data.datasets.base import GraphDataset


# TODO replace `xxx` class with its actual name when implemented
class PymatgenDataset(GraphDataset):
    """A dataset for loading structures from pymatgen. This class is should not be used for large
    datasets, as it need to detect the space group number for each structure in the dataset, which
    may take a significant amount of time. To load an already preprocessed dataset, use the `xxx`
    class instead.

    Args:
        root (str): Root directory of the dataset.
        transform (Callable | None): A function/transform that takes in a graph and returns a transformed version.
        struct_transform (Callable | None): A function/transform that takes in a structure and returns a transformed version.
        target_transform (Callable | None): A function/transform that takes in a target and returns a transformed version.
        symprec (float): Tolerance for space group detection. (default: 0.01)
        **graph_kwargs: Additional keyword arguments to be passed to the Graph class.

    Methods:
        __getitem__: Retrieves a graph and its corresponding target from the dataset.
        __len__: Returns the length of the dataset.
        load: Loads the data from the resource files.
        download: Downloads the Aflow dataset if it doesn't exist already.
    """

    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        struct_transform: Callable | None = None,
        target_transform: Callable | None = None,
        symprec: float = 0.01,
        graph_kwargs: dict[str, Any] = {},
    ) -> None:
        super().__init__(root, transform, struct_transform, target_transform, graph_kwargs)

        self.data = self.load_data()
        self.targets = self.load_targets(self.data, symprec)
        self.classes = list(set(self.targets))

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        data, target = self.data[index], self.targets[index]

        struct = Structure.from_file(data)
        if self.struct_transform is not None:
            struct = self.struct_transform(struct)

        graph = self.knn.convert(struct)
        if self.transform is not None:
            graph: Data = self.transform(graph)

        if self.target_transform is not None:
            target: int = self.target_transform(target)

        return graph, target

    def __len__(self) -> int:
        return len(self.data)

    def load_data(self) -> list[Path]:
        data = [file for file in self.raw_folder.iterdir()]
        return data

    def load_targets(self, data: list[Path], symprec: float) -> list[int]:
        targets = []
        for file in data:
            struct = Structure.from_file(file)
            sga = SpacegroupAnalyzer(struct, symprec=symprec)
            targets.append(sga.get_space_group_number())
        return targets
