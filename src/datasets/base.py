from collections.abc import Callable

from torch_geometric.data import InMemoryDataset


class Dataset(InMemoryDataset):
    """Base dataset class for graph data.
    This class has two main purposes:
    1. To provide a common interface for all datasets in the project.
    2. Restrict the CLI from thinking that all PyG datasets can be used in the project.
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
