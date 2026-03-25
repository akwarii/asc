import inspect
import json
import os.path as osp
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from src.datasets.base import Dataset


class MaterialProject(Dataset):
    """A dataset class for the Material Project dataset.

    Args:
        root: Root directory of the dataset.
        transform: A function that takes in a graph and returns a transformed version.
        pre_transform: A function that takes in a graph and returns a transformed version.
        pre_filter: A function that takes in a graph and returns a boolean value indicating
            whether the graph should be included in the dataset.
        force_reload: Whether to reload the dataset even if it already exists.
        download_only: Whether to download the dataset only without processing and loading it.
        search_kwargs: Keyword arguments forwarded to `mpr.materials.summary.search`.
        kwargs: Additional keyword arguments to be passed to PeriodicKNN or Dataset class.
    """

    DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"
    DOTENV_KEY = "MATERIALS_PROJECT_API_KEY"

    def __init__(
        self,
        root: str = "data/material_project",
        *,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        force_reload: bool = False,
        download_only: bool = False,
        search_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        self.search_kwargs = dict(search_kwargs or {})

        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            force_reload=force_reload,
            download_only=download_only,
            **kwargs,
        )

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory."""
        return osp.join(self.root, "processed", f"{self.kwargs['k']}nn")

    @property
    def raw_file_names(self) -> list[str]:
        """Return the name of the downloaded files."""
        return [f"data_{class_idx}.json" for class_idx in range(1, 231)]

    @property
    def processed_file_names(self) -> list[str]:
        """Return the name of the processed files ie the transformed data saved to the disk."""
        return ["data.pt"]

    @staticmethod
    def _validate_search_kwargs(
        search_callable: Callable,
        candidate_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate kwargs against the MP summary search signature."""
        reserved = {"spacegroup_number", "fields"}
        conflict = sorted(reserved.intersection(candidate_kwargs))
        #! Note: Maybe we can think of allowing users to specify spacegroups ?
        if conflict:
            raise ValueError(
                "`spacegroup_number` and `fields` are managed internally and should not be set "
                f"in search kwargs: {conflict}."
            )

        signature = inspect.signature(search_callable)
        accepted = set(signature.parameters)
        unsupported = sorted(set(candidate_kwargs) - accepted)
        if unsupported:
            raise ValueError(f"Unsupported Material Project search kwargs: {unsupported}.")

        return {key: candidate_kwargs[key] for key in candidate_kwargs if key in accepted}

    def download(self) -> None:
        """Download the dataset from Material Project and store it in the raw directory."""
        from dotenv import get_key

        try:
            from mp_api.client import MPRester
            from pymatgen.core.structure import Structure
        except ImportError:
            raise ImportError(
                "The Materials Project API client is not installed. "
                "Install it with `pip install mp-api`."
            )

        api_key = get_key(self.DOTENV_PATH, self.DOTENV_KEY)
        api = MPRester(api_key, mute_progress_bars=True, use_document_model=False)

        with api as mpr:
            search_kwargs = self._validate_search_kwargs(
                mpr.materials.summary.search,
                self.search_kwargs,
            )

            for idx in tqdm(range(1, 231)):
                docs = mpr.materials.summary.search(
                    spacegroup_number=idx,
                    fields=[
                        "symmetry",
                        "structure",
                        "deprecated",
                        "warnings",
                    ],
                    **search_kwargs,
                )

            # Filter out deprecated and warning entries and convert to POSCAR format
            # Pymatgen throws UserWarning when electronegativity is not found, we can ignore it
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                filtered_data = [
                    {
                        "structure": Structure.from_dict(entry["structure"]).to(fmt="poscar"),
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

        from src.graph import PeriodicKNN

        if self.download_only:
            return

        raw_data_list, target_list = [], []
        for file in self.raw_paths:
            with open(file) as json_file:
                json_data = json.load(json_file)
            raw_data_list += [entry["structure"] for entry in json_data]
            target_list += [entry["spacegroup"] for entry in json_data]

        # Convert the target labels to consecutive 0-based indices
        unique_targets = sorted(set(target_list))
        label_to_index = {label: idx for idx, label in enumerate(unique_targets)}
        target_list = [label_to_index[target] for target in target_list]

        knn = PeriodicKNN(**self.kwargs)

        data_list = []
        for raw_data, target in tqdm(zip(raw_data_list, target_list), total=len(raw_data_list)):
            data = knn.convert(raw_data)

            if data.num_nodes is None or data.num_nodes == 0:
                raise RuntimeError("The number of nodes in the graph is zero.")

            data.y = torch.full((data.num_nodes,), target, dtype=torch.long)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
