from collections.abc import Callable
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


# TODO add a graph factory attribute and needed kargs in the __init__ method
class CrystalGraphDataset(Dataset):
    """Base class for making datasets which are compatible with crystal graphs. It is necessary to
    override the ``__getitem__`` and ``__len__`` method. A ``download`` method can also be
    implemented to download the dataset. (This class implementation is based on the torchvision
    VisionDataset class)

    Args:
        root (string): Root directory of dataset.
        transform (Callable | None): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (Callable | None): A function/transform that takes in the
            target and transforms it.
    """

    _repr_indent = 4

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        struct_transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        if isinstance(root, str):
            root = Path(root)
        self.root = root

        self.transform = transform
        self.struct_transform = struct_transform
        self.target_transform = target_transform

        self.transforms = StandardTransform(transform, struct_transform, target_transform)

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
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += self.extra_repr().splitlines()
        if hasattr(self, "transform") and self.transform is not None:
            body += [repr(self.transform)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    @property
    def raw_folder(self) -> str:
        return self.root / self.__class__.__name__ / "raw"

    @property
    def processed_folder(self) -> str:
        return self.root / self.__class__.__name__ / "processed"

    def _format_transform_repr(self, transform: Callable, head: str) -> list[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def download(self) -> None:
        """Download the dataset if it doesn't exist."""
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
