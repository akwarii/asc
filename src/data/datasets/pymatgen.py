from typing import Any

from src.data.datasets.base import GraphDataset


# TODO implementation of a dataset class that loads data from a custom source
# data must be readable by pymatgen,
# This class is mostly used to load small datasets that can fit in memory. The dataset
# is most likely created using the `src.processing.KNNGraph.convert_and_save` method
# and loaded here using the `src.data.datasets.in_memory.InMemoryDataset.load` method.
class PymatgenDataset(GraphDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        return super().__getitem__(index)

    def __len__(self) -> int:
        return super().__len__()

    def load(self, path) -> None:
        pass
