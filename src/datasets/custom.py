from collections.abc import Callable
from pathlib import Path
from typing import Any

from torch_geometric.data import InMemoryDataset


class CustomDataset(InMemoryDataset):
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

    def __init__(
        self,
        root: str = "data/custom",
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
        import warnings

        import torch
        from pymatgen.io.vasp.inputs import BadPoscarWarning
        from tqdm.auto import tqdm

        from src.graph import KNNGraph

        raw_data_list, target_list = [], []
        for file in self.raw_paths:
            with open(file) as f:
                lines = f.readlines()
            raw_data_list.append("".join(lines))
            target_list.append(int(lines[0].strip().split()[0][0]))

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
