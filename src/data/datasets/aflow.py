import json
from pathlib import Path
from typing import Any, Callable, Optional

from pymatgen.core import Structure
from tqdm.auto import tqdm

from src.api.aflow import AflowAPI
from src.data.datasets.base_dataset import CrystalGraphDataset
from src.processing.graph import Graph


class Aflow(CrystalGraphDataset):
    """
    A dataset class for the Aflow dataset.

    Args:
        root (str): Root directory of the dataset.
        transform (Optional[Callable]): A function/transform that takes in a graph and returns a transformed version.
        struct_transform (Optional[Callable]): A function/transform that takes in a structure and returns a transformed version.
        target_transform (Optional[Callable]): A function/transform that takes in a target and returns a transformed version.
        download (bool): Whether to download the dataset if it doesn't exist.
        chunk_size (int): Number of entries of each chunk to download.
        **graph_kwargs: Additional keyword arguments to be passed to the Graph class.

    Attributes:
        api (AflowAPI): An instance of the AflowAPI class.
        classes (list): A list of space group numbers.
        resources (list): A list of resource filenames.

    Methods:
        __init__: Initializes the Aflow dataset.
        __getitem__: Retrieves a graph and its corresponding target from the dataset.
        __len__: Returns the length of the dataset.
        raw_folder: Returns the path to the raw folder.
        processed_folder: Returns the path to the processed folder.
        _load_data: Loads the data from the resource files.
        _check_exists: Checks if the dataset files exist.
        download: Downloads the Aflow dataset if it doesn't exist already.
    """

    api = AflowAPI()

    classes = list(range(1, 231))  # space groups numbers

    resources = [f"data_{class_idx}.json" for class_idx in classes]

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        struct_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        chunk_size: int = 100_000,
        **graph_kwargs,
    ) -> None:
        super().__init__(root, transform, struct_transform, target_transform)
        self.graph_kwargs = graph_kwargs

        if download:
            self.download(chunk_size)

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self.data, self.targets = self._load_data()

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        contcar, target = self.data[index], self.targets[index]

        struct = Structure.from_str(contcar, fmt="poscar")

        if self.struct_transform is not None:
            struct = self.struct_transform(struct)

        # TODO: really need to refactor Graph to a graph factory to improve efficiency
        # and if possible use DGL/PyG graphs instead of custom implementation
        graph = Graph(**self.graph_kwargs).set_features(struct)

        if self.transform is not None:
            graph = self.transform(graph)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return graph, target

    def __len__(self) -> int:
        return len(self.data)

    def _load_data(self) -> tuple[list[str], list[int]]:
        files = [Path(self.raw_folder, fname) for fname in self.resources]

        data, targets = [], []
        for file in files:
            with file.open("r") as json_file:
                json_data = json.load(json_file)
            data += [entry["CONTCAR.relax"] for entry in json_data]
            targets += [entry["spacegroup_relax"] for entry in json_data]

        return data, targets

    def _check_exists(self) -> bool:
        return all(Path(self.raw_folder, fname).is_file() for fname in self.resources)

    def download(self, chunk_size: int) -> None:
        """
        Downloads the Aflow dataset if it doesn't exist already.

        Args:
            chunk_size (int): Number of entries of each chunk to download.
        """
        if self._check_exists():
            return

        Path(self.raw_folder).mkdir(parents=True, exist_ok=True)

        print(f"Downloading Aflow data from {self.api.base_url} to {self.raw_folder}...")
        for class_idx in tqdm(self.classes):
            file = Path(self.raw_folder, f"data_{class_idx}.json")

            if file.is_file() and file.stat().st_size > 0:
                continue

            # Download data by chunks to avoid server timeout
            page_number = 1
            total_data = []

            with self.api as aflow_api:
                while True:
                    current_data = aflow_api.request(
                        f"spacegroup_relax({class_idx})",
                        paging_range=(page_number, chunk_size),
                    )
                    if not current_data:
                        break
                    page_number += 1
                    total_data.extend(current_data)

                compounds = set()
                filtered_data = []
                for entry in total_data:
                    if entry["compound"] not in compounds:
                        try:
                            entry["CONTCAR.relax"] = aflow_api.get_contcar(entry)
                        except RuntimeError:
                            continue
                        compounds.add(entry["compound"])
                        del entry["Pearson_symbol_relax"], entry["compound"]
                        filtered_data.append(entry)

            with open(file, "w") as f:
                json.dump(filtered_data, f, sort_keys=True, indent=4)
