import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pymatgen.core import Structure
from tqdm.auto import tqdm

from src.constants import CUSTOM_CLASSES
from src.datasets.base import GraphDataset


# TODO this class needs to be refactored to be more generic and better
# integrated with the rest of the code
# TODO docstring in google format
class CustomDataset(GraphDataset):
    """A dataset class for any user provided custom dataset.

    Args:
        root: Root directory of the dataset.
        transform: A function that takes in a graph and returns a transformed version.
        struct_transform: A function that takes in a structure and returns a transformed version.
        target_transform: A function that takes in a target and returns a transformed version.
        fetch_data: Whether we need to pretreat raw data if the dataset doesn't exist.
        origin_dir: Directory containing the raw files (in non-json format)
        graph_kwargs: Additional keyword arguments to be passed to the Graph class.

    Attributes:
        classes (list): a list of space group numbers ranking from 1 to 230.
        resources (list): Names of the files containing the dataset.
    """

    classes = CUSTOM_CLASSES

    resources = tuple([f"data_{class_idx}.json" for class_idx in classes])

    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        struct_transform: Callable | None = None,
        target_transform: Callable | None = None,
        fetch_data: bool = False,
        origin_dir: str | None = None,
        graph_kwargs: dict[str, Any] = {},  # TODO never use a mutable default argument
        **kwargs: Any,
    ) -> None:
        if origin_dir is None:
            raise ValueError("origin_dir must be provided.")

        super().__init__(root, transform, struct_transform, target_transform, graph_kwargs)

        if fetch_data:
            self.fetch_data(origin_dir)

        if not self.check_exists():
            raise RuntimeError(
                "Dataset not found. Provide it with fetch_data=True and origin_dir=<path>"
            )

        self.data, self.targets = self.load()

    def __getitem__(self, index: int) -> tuple[Any, Any]:
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

    # TODO if the processed data is already available, we can load it instead
    # TODO doing so will save time during __getitem__ calls as we won't have
    # to convert the structure to a graph
    def load(self) -> tuple[list[str], list[int]]:
        """Load data from raw files and return a tuple of data and targets.

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
