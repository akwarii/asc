from collections.abc import Callable

from torch_geometric.data import InMemoryDataset


class Dataset(InMemoryDataset):
    """Base dataset class for graph data.
    This class has two main purposes:
    1. To provide a common interface for all datasets in the project.
    2. Restrict the CLI from thinking that all PyG datasets can be used in the project.

    Args:
        root: Root directory of the dataset.
        transform: A function that takes in a graph and returns a transformed version.
        pre_transform: A function that takes in a graph and returns a transformed version.
        pre_filter: A function that takes in a graph and returns a boolean value indicating
            whether the graph should be included in the dataset.
        force_reload: Whether to reload the dataset even if it already exists.
        download_only: Whether to download the dataset only without processing and loading it.
        kwargs: Additional keyword arguments to be passed to PeriodicKNN or Dataset class.
    """

    def __init__(
        self,
        root: str | None = None,
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
