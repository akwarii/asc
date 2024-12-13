import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pymatgen.core import Structure
from tqdm.auto import tqdm

from src.data.datasets.base import GraphDataset
from src.utils.constants import CUSTOM_CLASSES


class CustomDataset(GraphDataset):
    """A dataset class for any user provided custom dataset.

    Args:
        root (str): Root directory of the dataset.
        transform (Callable | None): A function/transform that takes in a graph and returns a transformed version.
        struct_transform (Callable | None): A function/transform that takes in a structure and returns a transformed version.
        target_transform (Callable | None): A function/transform that takes in a target and returns a transformed version.
        fetch_data (bool): whether we need to find and pretreat raw data beforehand if the dataset doesn't exist.
        origin_dir (str | None): Directory containing the raw files (in non-json format)
        graph_kwargs: Additional keyword arguments to be passed to the Graph class.

    Attributes:
        classes (list): a list of space group numbers ranking from 1 to 230.
        resources (list): Names of the files containing the dataset.

    Methods:
        __getitem__: Retrieves a graph and its corresponding target from the dataset.
        __len__: Returns the length of the dataset.
        load: Loads the data from the resource files.
        fetch_data: From an existing local directory with unformatted (ie. non-json) labelled structures, extract and pretreats the content if the database doesn't exist.
    """

    classes = CUSTOM_CLASSES

    resources = [f"data_{class_idx}.json" for class_idx in classes]

    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        struct_transform: Callable | None = None,
        target_transform: Callable | None = None,
        fetch_data: bool = False,
        origin_dir: str | None = None,
        graph_kwargs: dict[str, Any] = {},
    ) -> None:
        super().__init__(root, transform, struct_transform, target_transform, graph_kwargs)

        if fetch_data:
            self.fetch_data(origin_dir)

        if not self.check_exists():
            raise RuntimeError(
                "Dataset not found. You can use fetch_data=True and origin_dir=<path> to provide it"
            )

        self.data, self.targets = self.load()

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        if not self.data:
            RuntimeWarning("Dataset not loaded. Use load=True to load the dataset")
            return None, None

        contcar, target = self.data[index], self.targets[index]

        struct = Structure.from_str(contcar, fmt="poscar")

        if self.struct_transform is not None:
            struct = self.struct_transform(struct)

        graph = self.knn.convert(struct)

        if self.transform is not None:
            graph = self.transform(graph)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return graph, target

    def __len__(self) -> int:
        return len(self.data)

    def load(self) -> tuple[list[str], list[int]]:
        """Load data from JSON files and return a tuple of data and targets.

        Returns:
            tuple[list[str], list[int]]: A tuple containing the loaded data and targets.
        """
        files = [self.raw_folder / fname for fname in self.resources]

        data, targets = [], []
        for file in files:
            with open(file) as json_file:
                json_data = json.load(json_file)
            data += [entry["structure"] for entry in json_data]
            targets += [entry["spacegroup"] for entry in json_data]

        return data, targets

    # TODO: write and test this
    # TODO GH optimize before pushing to production
    # FIXME GH incorrect docstring
    def fetch_data(self, origin_dir: str) -> None:
        """Downloads the Aflow dataset if it doesn't exist already."""
        if self.check_exists():
            print(f"Dataset already exists at {self.root}")
            return

        self.raw_folder.mkdir(parents=True, exist_ok=True)

        print(f"Converting custom data data from {origin_dir} to {self.raw_folder}...")

        # Might be slow because each file has to be opened "self.classes" times
        for idx in tqdm(self.classes):
            file = self.raw_folder / f"data_{idx}.json"

            filtered_data = []

            for path in Path(origin_dir).rglob("*.POSCAR"):
                with open(path) as poscar:
                    lines = poscar.readlines()
                    if lines[0] == str(idx) + "\n":
                        filtered_data.append(
                            {
                                "structure": "".join(lines),
                                "spacegroup": idx + 1,
                            }
                        )

            with open(file, "w") as f:
                json.dump(filtered_data, f, sort_keys=True, indent=4)
