from collections.abc import Callable
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.graph import KNNGraph
from src.constants import REPR_INDENT
from src.typing import PathLike


class GraphDataset(Dataset):
    """Base class for making datasets which are compatible with crystal graphs. It is necessary to
    override the ``__getitem__``, ``__len__`` and ``load`` method. A ``fetch_data`` method can also
    be implemented to get the dataset. (This class implementation is based on the torchvision
    VisionDataset class)

    Args:
        root (string): Root directory of dataset.
        transform (Callable | None): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (Callable | None): A function/transform that takes in the
            target and transforms it.
    """

    resources = []

    def __init__(
        self,
        root: PathLike,
        transform: Callable | None = None,
        struct_transform: Callable | None = None,
        target_transform: Callable | None = None,
        graph_kwargs: dict[str, Any] = {},
    ) -> None:
        if isinstance(root, str):
            root = Path(root)
        self.root = root

        self.transform = transform
        self.struct_transform = struct_transform
        self.target_transform = target_transform

        self.transforms = StandardTransform(transform, struct_transform, target_transform)

        self.knn = KNNGraph(**graph_kwargs)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """
        Args:
            index (int): Index

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
        body += self.extra_repr().splitlines()
        if hasattr(self, "transform") and self.transform is not None:
            body += [repr(self.transform)]
        lines = [head] + [" " * REPR_INDENT + line for line in body]
        return "\n".join(lines)

    @property
    def raw_folder(self) -> Path:
        return self.root / self.__class__.__name__.lower() / "raw"

    @property
    def processed_folder(self) -> Path:
        return self.root / self.__class__.__name__.lower() / "processed"

    def _format_transform_repr(self, transform: Callable, head: str) -> list[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def check_exists(self) -> bool:
        """Check if every file in the resources attribute exists in the raw folder."""
        return all((self.raw_folder / fname).is_file() for fname in self.resources)

    def fetch_data(self) -> None:
        """Fetch the dataset if it doesn't exist."""
        raise NotImplementedError

    def load(self) -> tuple[list[str], list[int]]:
        """Load data from resources and return a tuple of data and targets.

        Returns:
            tuple[list[str], list[int]]: A tuple containing the loaded data and targets.
        """
        raise NotImplementedError

    def extra_repr(self) -> str:
        return ""


class StandardTransform:
    def __init__(
        self,
        transform: Callable | None = None,
        struct_transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        self.transform = transform
        self.struct_transform = struct_transform
        self.target_transform = target_transform

    def __call__(self, input: Any, struct: Any, target: Any) -> tuple[Any, Any]:
        if self.transform is not None:
            input = self.transform(input)
        if self.struct_transform is not None:
            struct = self.struct_transform(struct)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform: Callable, head: str) -> list[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform, "Transform: ")
        if self.struct_transform is not None:
            body += self._format_transform_repr(self.struct_transform, "Struct transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform, "Target transform: ")

        return "\n".join(body)
