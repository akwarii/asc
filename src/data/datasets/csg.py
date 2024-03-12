from collections.abc import Callable
from typing import Any

import pandas as pd
from kaggle import KaggleApi
from pymatgen.core import Structure

from src.data.datasets.base import GraphDataset
from src.data.datasets.utils import check_integrity
from src.processing.graph import KNNGraph


class CSG(GraphDataset):
    """The Crystal Space Group (CSG) dataset is a preprocessed version of the AFLOW, GNoME and Material Project datasets.
    The dataset contains ~1,050,000 crystal structures with space group numbers ranging from 1 to 230. The dataset is
    formatted as a CSV file with two columns: "SpaceGroupNumber" and "Structure". The "Structure" column contains the
    string representation of the crystal structure in the POSCAR format.

    The AFLOW data was filtered to only include structures with a maximum stress component of +/-0.1 GPa and a maximum
    force component of +/-0.01 eV/A.
    Material Project data was filtered to remove structures with deprecated or warning flags.
    All GNoME data predicted stable were included.
    Additionally, structures with both the same space group number and composition were removed to avoid redundancy.

    The maximum number of atoms in a structure is 444. A radius graph with a cutoff of 10 angstroms ensure that 99.5%
    of the graphs are complete.

    The dataset is available for download from Kaggle at https://www.kaggle.com/datasets/gaelhuynh/space-group.

    Args:
        root (str): Root directory of the dataset.
        transform (Callable | None): A function/transform that takes in a graph and returns a transformed version.
        struct_transform (Callable | None): A function/transform that takes in a structure and returns a transformed version.
        target_transform (Callable | None): A function/transform that takes in a target and returns a transformed version.
        download (bool): Whether to download the dataset if it doesn't exist.
        load (bool): Whether to load the dataset.
        **graph_kwargs: Additional keyword arguments to be passed to the Graph class.

    Attributes:
        classes (list): A list of space group numbers ranging from 1 to 230.
        resources (list): Names of the files containing the dataset.

    Methods:
        __init__: Initializes the Aflow dataset.
        __getitem__: Retrieves a graph and its corresponding target from the dataset.
        __len__: Returns the length of the dataset.
        _load_data: Loads the data from the resource files.
        download: Downloads the dataset if it doesn't exist already.
    """

    KAGGLE_DATASET = "gaelhuynh/space-group"
    API = KaggleApi()

    classes = list(range(1, 231))  # space groups numbers

    resources = ("CSG.csv",)
    md5_checksums = ("685236d6e7fd6677d3dd809897ecb393",)

    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        struct_transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
        load: bool = True,
        **graph_kwargs,
    ) -> None:
        super().__init__(root, transform, struct_transform, target_transform)
        self.graph_kwargs = graph_kwargs

        if download:
            self.download()

        if not self.check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        if load:
            self.data, self.targets = self._load_data()

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        if not self.data:
            RuntimeWarning("Dataset not loaded. Use load=True to load the dataset")
            return None, None

        contcar, target = self.data[index], self.targets[index]

        struct = Structure.from_str(contcar, fmt="poscar")

        if self.struct_transform is not None:
            struct = self.struct_transform(struct)

        # TODO: really need to refactor Graph to a graph factory to improve efficiency
        # and if possible use DGL/PyG graphs instead of custom implementation
        graph = KNNGraph(**self.graph_kwargs)
        graph.set_features(struct)

        if self.transform is not None:
            graph = self.transform(graph)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return graph, target

    def __len__(self) -> int:
        return len(self.data)

    def _load_data(self) -> tuple[list[str], list[int]]:
        """Load data from JSON files and return a tuple of data and targets.

        Returns:
            tuple[list[str], list[int]]: A tuple containing the loaded data and targets.
        """
        df = pd.read_csv(self.raw_folder / self.resources[0])
        data = df["Structure"].tolist()
        targets = df["SpaceGroupNumber"].tolist()

        return data, targets

    def download(self) -> None:
        """Downloads the Aflow dataset if it doesn't exist already."""

        if self.check_exists():
            return

        self.API.authenticate()
        self.API.dataset_download_files(
            self.KAGGLE_DATASET, path=self.raw_folder, quiet=False, unzip=True
        )

        paths = [self.raw_folder / resource for resource in self.resources]
        check_integrity(paths, self.md5_checksums)
