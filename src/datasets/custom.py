from collections.abc import Callable
from pathlib import Path

from torch_geometric.data import InMemoryDataset


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
        kwargs: Additional keyword arguments to be passed to the KNNGraph or InMemoryDataset class.
    """

    def __init__(
        self,
        root: str = "data/custom",
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
        kwargs.pop("rcut", None)
        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload, **kwargs
        )

        if not self.download_only:
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        """Return the name of the downloaded files."""
        return list([f.name for f in Path(self.raw_dir).rglob("*")])

    @property
    def processed_file_names(self) -> list[str]:
        """Return the name of the processed files ie the transformed data saved to the disk."""
        return [f"data_{self.kwargs['k']}nn.pt"]

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

        if self.download_only:
            return

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

            data.y = torch.full((data.num_nodes,), target, dtype=torch.long)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
