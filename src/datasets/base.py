from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.constants import REPR_INDENT
from src.graph import KNNGraph
from src.typing import PathLike


class GraphDataset(Dataset):
    """Base class for making datasets which are compatible with crystal graphs.

    It is necessary to override the ``__getitem__``, ``__len__`` and ``load`` method. A
    ``fetch_data`` method can also be implemented to get the dataset. (This class implementation
    is based on the torchvision VisionDataset class)

    Attributes:
        root (string): Root directory of dataset.
        transform (Callable | None): A function/transform that takes in an input and returns a
            transformed version.
        struct_transform (Callable | None): A function/transform that takes in a structure and
            returns a transformed version.
        target_transform (Callable | None): A function/transform that takes in a target and
            returns a transformed version.
        knn (KNNGraph): A class to generate the k-nearest neighbor graph.
    """

    resources: tuple[str, ...] = ()

    def __init__(
        self,
        root: PathLike,
        transform: Callable | None = None,
        struct_transform: Callable | None = None,
        target_transform: Callable | None = None,
        graph_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initializes a dataset, its transforms and the graph generation.

        Args:
            root: Root directory of the dataset.
            transform: A function/transform that takes in an input and returns a transformed
                version.
            struct_transform: A function/transform that takes in a structure and returns a
                transformed version.
            target_transform: A function/transform that takes in a target and returns a
                transformed version.
            graph_kwargs: Additional keyword arguments to be passed to the KNNGraph class.
        """
        if isinstance(root, str):
            root = Path(root)
        self.root = root

        self.transform = transform
        self.struct_transform = struct_transform
        self.target_transform = target_transform

        if graph_kwargs is None:
            graph_kwargs = {}
        self.knn = KNNGraph(**graph_kwargs)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Args:
            index: Index of the sample to be loaded.

        Returns:
            (Any): Sample and meta data, optionally transformed by the respective transforms.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = f"Dataset {self.__class__.__name__}"
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        if hasattr(self, "transform") and self.transform is not None:
            body += [repr(self.transform)]
        lines = [head] + [" " * REPR_INDENT + line for line in body]
        return "\n".join(lines)

    @property
    def raw_folder(self) -> Path:
        """Folder containing the raw data, such as csv or json."""
        return self.root / self.__class__.__name__.lower() / "raw"

    @property
    def processed_folder(self) -> Path:
        """Folder containing the processed data, such as PyG Data or torch Tensor."""
        return self.root / self.__class__.__name__.lower() / "processed"

    def _format_transform_repr(self, transform: Callable, head: str) -> list[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def check_exists(self) -> bool:
        """Check if every file in the resources attribute exists in the raw folder."""
        return all((self.raw_folder / fname).is_file() for fname in self.resources)

    @abstractmethod
    def load(self) -> tuple[list[str], list[int]]:
        """Load data from raw files and return a tuple of data and targets.

        Returns:
            tuple[list[str], list[int]]: A tuple containing the loaded data and targets.
        """
        raise NotImplementedError

    # TODO mypy is complaining because the signature changes in the subclasses
    # @abstractmethod
    # def fetch_data(self) -> None:
    #     """Download the dataset from the source."""
    #     raise NotImplementedError
