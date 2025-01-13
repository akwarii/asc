import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from torch_geometric.data import InMemoryDataset
from tqdm.auto import tqdm


class MaterialProject(InMemoryDataset):
    """A dataset class for the Material Project dataset.

    Args:
        root: Root directory of the dataset.
        transform: A function that takes in a graph and returns a transformed version.
        pre_transform: A function that takes in a graph and returns a transformed version.
        pre_filter: A function that takes in a graph and returns a boolean value indicating
            whether the graph should be included in the dataset.
        force_reload: Whether to reload the dataset even if it already exists.
        kwargs: Additional keyword arguments to be passed to the KNNGraph or InMemoryDataset class.
    """

    DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"
    DOTENV_KEY = "MATERIALS_PROJECT_API_KEY"

    def __init__(
        self,
        root: str = "data/material_project",
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        force_reload: bool = False,
        **kwargs: Any,
    ) -> None:
        self.kwargs = kwargs

        kwargs.pop("k", None)
        kwargs.pop("rcut", None)
        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload, **kwargs
        )

        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        """Return the name of the downloaded files."""
        return [f"data_{class_idx}.json" for class_idx in range(1, 231)]

    @property
    def processed_file_names(self) -> list[str]:
        """Return the name of the processed files ie the transformed data saved to the disk."""
        return ["data.pt"]

    def download(self) -> None:
        """Download the dataset from Material Project and store it in the raw directory."""
        from dotenv import get_key

        try:
            from mp_api.client import MPRester
        except ImportError:
            raise ImportError(
                "The Materials Project API client is not installed. "
                "Install it with `pip install mp-api`."
            )

        api_key = get_key(self.DOTENV_PATH, self.DOTENV_KEY)
        api = MPRester(api_key, mute_progress_bars=True, use_document_model=False)

        for idx in tqdm(range(1, 231)):
            with api as mpr:
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

            file = f"{self.raw_dir}/data_{idx}.json"
            try:
                with open(file, "w") as f:
                    json.dump(filtered_data, f, sort_keys=True, indent=4)
            except OSError as e:
                print(f"Error writing data to file {file}: {e}")

    def process(self) -> None:
        """Process the dataset by converting the structures to graphs, applying both pre-filter and
        pre-transform functions, and saving the processed data to disk. The data is saved in the
        processed directory as a single file named "data.pt".
        """
        import torch
        from pymatgen.io.vasp.inputs import BadPoscarWarning

        from src.graph import KNNGraph

        raw_data_list, target_list = [], []
        for file in self.raw_paths:
            with open(file) as json_file:
                json_data = json.load(json_file)
            raw_data_list += [entry["structure"] for entry in json_data]
            target_list += [entry["spacegroup"] for entry in json_data]

        knn = KNNGraph(**self.kwargs)

        data_list = []
        for raw_data, target in tqdm(zip(raw_data_list, target_list), total=len(raw_data_list)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", BadPoscarWarning)
                data = knn.convert(raw_data)

            if data.num_nodes is None or data.num_nodes == 0:
                raise RuntimeError("The number of nodes in the graph is zero.")

            data.y = torch.full((data.num_nodes,), target - 1, dtype=torch.long)

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
