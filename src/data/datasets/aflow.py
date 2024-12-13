import json
from collections.abc import Callable
from typing import Any

import numpy as np
from pymatgen.core import Structure
from torch_geometric.data import Data
from tqdm.auto import tqdm

from src.api import AflowAPI
from src.data.datasets.base import GraphDataset
from src.data.datasets.utils import poscar_from_entry
from src.utils.constants import AFLOW_CLASSES


class Aflow(GraphDataset):
    """A dataset class for the Aflow dataset.

    Args:
        root (str): Root directory of the dataset.
        transform (Callable | None): A function/transform that takes in a graph and returns a transformed version.
        struct_transform (Callable | None): A function/transform that takes in a structure and returns a transformed version.
        target_transform (Callable | None): A function/transform that takes in a target and returns a transformed version.
        fetch_data (bool): Whether to download the dataset if it doesn't exist.
        chunk_size (int): Number of entries of each chunk to download.
        **graph_kwargs: Additional keyword arguments to be passed to the Graph class.

    Attributes:
        API (AflowAPI): An instance of the AflowAPI class.
        classes (list): A list of space group numbers ranging from 1 to 230.
        resources (list): Names of the files containing the dataset.

    Methods:
        __getitem__: Retrieves a graph and its corresponding target from the dataset.
        __len__: Returns the length of the dataset.
        load: Loads the data from the resource files.
        fetch_data: Downloads the Aflow dataset if it doesn't exist already.
    """

    API = AflowAPI()
    classes = AFLOW_CLASSES

    resources = [f"data_{class_idx}.json" for class_idx in classes]

    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        struct_transform: Callable | None = None,
        target_transform: Callable | None = None,
        fetch_data: bool = False,
        chunk_size: int = 50_000,
        stress_threshold: int | None = None,
        force_threshold: float | None = None,
        graph_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> None:
        super().__init__(root, transform, struct_transform, target_transform, graph_kwargs)

        if fetch_data:
            self.fetch_data(chunk_size, stress_threshold, force_threshold)

        if not self.check_exists():
            raise RuntimeError("Dataset not found. You can use fetch_data=True to download it")

        self.data, self.targets = self.load()

    def __getitem__(self, index: int) -> tuple[Data, int]:
        data, target = self.data[index], self.targets[index]

        struct = Structure.from_str(data, fmt="poscar")
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

    def load(self) -> tuple[list[str], list[int]]:
        files = [self.raw_folder / fname for fname in self.resources]

        data, targets = [], []
        for file in files:
            with file.open("r") as json_file:
                json_data = json.load(json_file)

            data += [entry["structure"] for entry in json_data]
            targets += [entry["spacegroup"] for entry in json_data]

        return data, targets

    def fetch_data(
        self,
        chunk_size: int,
        stress_threshold: int | None = None,
        force_threshold: float | None = None,
    ) -> None:
        """Downloads the dataset if it doesn't exist already.

        Args:
            chunk_size (int): Number of entries per chunk.
            stress_threshold (int | None): Absolute stress threshold to filter the data.
            force_threshold (float | None): Absolute force threshold to filter the data.
        """
        if self.check_exists():
            print(f"Dataset already exists at {self.root}")
            return

        self.raw_folder.mkdir(parents=True, exist_ok=True)

        print(f"Downloading Aflow data from {self.API.base_url} to {self.raw_folder}...")
        for class_idx in tqdm(self.classes):
            file = self.raw_folder / f"data_{class_idx}.json"

            # Download data by chunks to avoid server timeout
            page_number = 1
            total_data = []
            matchbook = f"spacegroup_relax({class_idx}),geometry,positions_fractional"
            if stress_threshold:
                matchbook += ",stress_tensor"
            if force_threshold:
                matchbook += ",forces"

            with self.API as aflow_api:
                while True:
                    current_data = aflow_api.request(
                        matchbook=matchbook,
                        paging=page_number,
                        chunk_size=chunk_size,
                    )
                    if not current_data:
                        break
                    page_number += 1
                    total_data.extend(current_data)

            # Generate a CONTCAR for each unique compound
            compounds = set()
            filtered_data = []
            for entry in total_data:
                if stress_threshold and np.max(np.abs(entry["stress_tensor"])) > stress_threshold:
                    continue
                if force_threshold and np.max(np.abs(entry["forces"])) > force_threshold:
                    continue

                if entry["compound"] not in compounds:
                    compounds.add(entry["compound"])
                    reduced_entry = {
                        "spacegroup": entry["spacegroup_relax"],
                        "structure": poscar_from_entry(entry),
                    }
                    filtered_data.append(reduced_entry)

            with open(file, "w") as f:
                json.dump(filtered_data, f, sort_keys=True, indent=4)
