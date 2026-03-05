import os.path as osp
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from torch_geometric.data import InMemoryDataset


def _load_from_csv(file: str) -> tuple[list[str], list[int]]:
    """Load the dataset from a CSV file."""
    df = pd.read_csv(file)

    unique_labels = sorted(set(df["SpaceGroupNumber"]))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    raw_data_list = [row["Structure"] for _, row in df.iterrows()]
    target_list = df["SpaceGroupNumber"].map(label_to_index).tolist()

    return raw_data_list, target_list


def _load_from_dump(file: str) -> tuple[list[str], list[int]]:
    """Load the dataset from a LAMMPS dump file."""
    with open(file) as f:
        lines = f.readlines()

    atom_lines = None
    raw_data_list, target_list = [], []
    for line in lines:
        if line[:11] == "ITEM: ATOMS":
            atom_lines = lines[lines.index(line) + 1 :]
            break

    if atom_lines is None:
        raise ValueError("No atom lines found in the dump file.")

    for atom_line in atom_lines:
        raw_data_list.append(atom_line.strip())
        target_list.append(int(atom_line.strip().split()[-1]))

    return raw_data_list, target_list


EXTENSION_TO_PARSER = {
    ".csv": _load_from_csv,
    ".dump": _load_from_dump,
}


def get_raw_data_and_targets(file: str) -> tuple[list[str], list[int]]:
    """Get the raw data and targets from a file."""
    file_extension = Path(file).suffix.lower()
    parser = EXTENSION_TO_PARSER.get(file_extension)
    if parser is None:
        raise NotImplementedError(
            f"Unsupported file format {file_extension}. "
            f"Only {', '.join(EXTENSION_TO_PARSER.keys())} files are supported."
        )
    return parser(file)


class CustomDataset(InMemoryDataset):
    """A dataset class for any user provided custom dataset.

    Args:
        root: Root directory of the dataset.
        transform: A function that takes in a graph and returns a transformed version.
        pre_transform: A function that takes in a graph and returns a transformed version.
        pre_filter: A function that takes in a graph and returns a boolean value indicating
            whether the graph should be included in the dataset.
        force_reload: Whether to reload the dataset even if it already exists.
        download_only: Whether to download the dataset only without processing and loading it.
        kwargs: Additional keyword arguments to be passed to PeriodicKNN or InMemoryDataset class.
    """

    def __init__(
        self,
        root: str = "data/custom",
        *,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        force_reload: bool = False,
        download_only: bool = False,
        **kwargs,
    ) -> None:
        self.download_only = download_only
        self.kwargs = kwargs.copy()

        kwargs.pop("k", None)
        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload, **kwargs
        )

        if not self.download_only:
            self.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory."""
        return osp.join(self.root, "processed", f"{self.kwargs['k']}nn")

    @property
    def raw_file_names(self) -> list[str]:
        """Return the name of the downloaded files."""
        return list([f.name for f in Path(self.raw_dir).rglob("*")])

    @property
    def processed_file_names(self) -> list[str]:
        """Return the name of the processed files ie the transformed data saved to the disk."""
        return ["data.pt"]

    def process(self) -> None:
        """Process the dataset by converting the structures to graphs, applying both pre-filter and
        pre-transform functions, and saving the processed data to disk. The data is saved in the
        processed directory as a single file named "data.pt".
        """
        import torch
        from tqdm.auto import tqdm

        from src.graph import PeriodicKNN

        if self.download_only:
            return

        if not self.raw_paths:
            raise RuntimeError(f"No data found in {self.raw_dir}.")

        raw_data_list, target_list = [], []
        for file in self.raw_paths:
            raw_data, raw_target = get_raw_data_and_targets(file)
            raw_data_list.extend(raw_data)
            target_list.extend(raw_target)

        knn = PeriodicKNN(**self.kwargs)

        data_list = []
        for raw_data, raw_target in tqdm(
            zip(raw_data_list, target_list), total=len(raw_data_list)
        ):
            data = knn.convert(raw_data)

            if data.num_nodes is None or data.num_nodes == 0:
                raise RuntimeError("The number of nodes in the graph is zero.")

            if isinstance(raw_target, int):
                data.y = torch.full((data.num_nodes,), raw_target, dtype=torch.long)
            elif isinstance(raw_target, list):
                data.y = torch.tensor(raw_target, dtype=torch.long)
            else:
                raise ValueError("Something went wrong with the target.")

            assert torch.all(data.y >= 0).item(), (
                "The target labels must be non-negative integers."
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
