import json
from collections.abc import Callable
from typing import Any

from torch_geometric.data import InMemoryDataset
from tqdm.auto import tqdm


class Aflow(InMemoryDataset):
    """A dataset class for the Aflow dataset.

    Args:
        root: Root directory of the dataset.
        transform: A function that takes in a graph and returns a transformed version.
        pre_transform: A function that takes in a graph and returns a transformed version.
        pre_filter: A function that takes in a graph and returns a boolean value indicating
            whether the graph should be included in the dataset.
        force_reload: Whether to reload the dataset even if it already exists.
        download_only: Whether to download the dataset only without processing and loading it.
        kwargs: Additional keyword arguments to be passed to the KNNGraph or InMemoryDataset class.
    """

    def __init__(
        self,
        root: str = "data/aflow",
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        force_reload: bool = False,
        download_only: bool = False,
        **kwargs: Any,
    ) -> None:
        self.download_only = download_only
        self.kwargs = kwargs.copy()

        kwargs.pop("k", None)
        kwargs.pop("rcut", None)
        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload, **kwargs
        )

        if not self.download_only:
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
        """Download the dataset from the AFLOW database and store it in the raw directory."""
        from src.api import AflowAPI
        from src.utils.lattice import poscar_from_entry

        for idx in tqdm(range(1, 231)):
            page_number = 1
            total_data = []
            matchbook = f"spacegroup_relax({idx}),geometry,positions_fractional"

            # Download data by chunks to avoid server timeout
            with AflowAPI() as api:
                while True:
                    current_data = api.request(
                        matchbook=matchbook,
                        paging=page_number,
                        chunk_size=10_000,
                    )
                    if not current_data:
                        break
                    page_number += 1
                    total_data.extend(current_data)

            # Generate a CONTCAR for each unique compound
            compounds = set()
            filtered_data = []
            for entry in total_data:
                if entry["compound"] not in compounds:
                    compounds.add(entry["compound"])
                    reduced_entry = {
                        "spacegroup": entry["spacegroup_relax"],
                        "structure": poscar_from_entry(entry),
                    }
                    filtered_data.append(reduced_entry)

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
        import warnings

        import torch
        from pymatgen.io.vasp.inputs import BadPoscarWarning

        from src.graph import KNNGraph

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

        knn = KNNGraph(**self.kwargs)

        data_list = []
        for raw_data, target in tqdm(zip(raw_data_list, target_list), total=len(raw_data_list)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", BadPoscarWarning)
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
