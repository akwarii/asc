import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import get_key
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import BadPoscarWarning
from tqdm.auto import tqdm

from src.constants import MP_CLASSES
from src.datasets.base import GraphDataset


class MaterialProject(GraphDataset):
    """A dataset class for the Material Project dataset.

    Args:
        root: Root directory of the dataset.
        transform: A function that takes in a graph and returns a transformed version.
        struct_transform: A function that takes in a structure and returns a transformed version.
        target_transform: A function that takes in a target and returns a transformed version.
        fetch_data: Whether to download the dataset if it doesn't exist.
        graph_kwargs: Additional keyword arguments to be passed to the Graph class.

    Attributes:
        API_KEY (str): The Materials Project API key.
        API (MPRester): An instance of the MPRester class.
        classes (list): A list of space group numbers ranging from 1 to 230.
        resources (list): Names of the files containing the dataset.
    """

    _dotenv_path = Path(__file__).resolve().parents[2] / ".env"
    _dotenv_key = "MATERIALS_PROJECT_API_KEY"

    API_KEY = get_key(_dotenv_path, _dotenv_key)
    API = MPRester(API_KEY, mute_progress_bars=True, use_document_model=False)

    classes = MP_CLASSES

    resources = tuple([f"data_{class_idx}.json" for class_idx in classes])

    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        struct_transform: Callable | None = None,
        target_transform: Callable | None = None,
        fetch_data: bool = False,
        graph_kwargs: dict[str, Any] = {},  # TODO never use a mutable default argument
        **kwargs: Any,
    ) -> None:
        super().__init__(root, transform, struct_transform, target_transform, graph_kwargs)

        if fetch_data:
            self.fetch_data()

        if not self.check_exists():
            raise RuntimeError("Dataset not found. You can use fetch_data=True to download it")

        self.data, self.targets = self.load()

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        contcar, target = self.data[index], self.targets[index]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BadPoscarWarning)
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

    def fetch_data(self) -> None:
        """Downloads the Aflow dataset if it doesn't exist already."""
        if self.check_exists():
            print(f"Dataset already exists at {self.root}")
            return

        self.raw_folder.mkdir(parents=True, exist_ok=True)

        print(
            f"Downloading Material Project data from {self.API.endpoint} to {self.raw_folder}..."
        )
        for idx in tqdm(self.classes):
            file = self.raw_folder / f"data_{idx}.json"

            with self.API as mpr:
                docs = mpr.materials.summary.search(
                    spacegroup_number=idx,
                    fields=[
                        "symmetry",
                        "structure",
                        "deprecated",
                        "warnings",
                    ],
                )

            # Filter out deprecated and warning entries and convert to POSCAR format
            # Pymatgen throws UserWarning when electronegativity is not found, we can ignore it
            # TODO Need to remove selective dynamics tags from POSCAR
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                filtered_data = [
                    {
                        "structure": entry["structure"].to(fmt="poscar"),  # type: ignore
                        "spacegroup": entry["symmetry"]["number"],  # type: ignore
                    }
                    for entry in docs
                    if not entry["deprecated"] and not entry["warnings"]  # type: ignore
                ]

            with open(file, "w") as f:
                json.dump(filtered_data, f, sort_keys=True, indent=4)
