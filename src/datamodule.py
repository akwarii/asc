import copy
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from lightning import LightningDataModule
from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader, ImbalancedSampler

from src import datasets
from src.datasets.base import Dataset
from src.transforms import TRAIN_ONLY_TRANSFORMS
from src.typing import Stage
from src.utils import random_split
from src.utils.builder import class_instantiator

DATASET_FACTORY: dict[str, Callable] = {
    "aflow": datasets.Aflow,
    "csg": datasets.CSG,
    "custom": datasets.CustomDataset,
    "gnome": datasets.Gnome,
    "mp": datasets.MaterialProject,
}


class LightningDataset(LightningDataModule):
    """A wrapper around LightningDataset that sets the batch size as an attribute after
    initialization. It is only used to have a direct access to the batch size in the datamodule,
    which is expected by the BatchSizeFinder callback of Lightning.

    Args:
        dataset: The dataset to use for training. If lengths are provided, the dataset is split
            into training, validation, and test datasets (default: `None`).
        dataset_name: The name of the dataset to use. It can be either `aflow`, `csg`, `custom`,
            `gnome`, or `mp`. If `dataset` is provided, this argument is ignored (default: `None`).
        lengths: The lengths of the training, validation, and test datasets. If only one value is
            provided, the dataset is sliced and only the training dataset is used. If two values
            are provided, the dataset is split into training and validation. If three values are
            provided, the dataset is split into training, validation, and test. If not provided,
            the whole dataset is used for training (default: `None`).
        pred_dataset: The dataset to use for prediction (default: `None`).
        use_imbalance_sampler: Whether to use the ImbalancedSampler to balance the dataset. Note
            that other sampler can be used by providing it in the `sampler` argument but can't be
            used at the same time (default: `False`).
        pre_filters: A function or a list of functions that takes in a `~torch_geometric.data.Data`
            object and returns a boolean value, indicating whether the data object should be
            included in the dataset (default: `None`).
        pre_transforms: A function or a list of functions that takes in a
            `~torch_geometric.data.Data` object and returns a transformed version. The data object
            will be transformed once before being saved to disk. Pre-transforms apply to every
            stage and should therefore be deterministic (default: `None`).
        transforms: A function or a list of functions that takes in a `torch_geometric.data.Data`
            object and returns a transformed version. Runtime transforms apply before every
            access. Registered train-only transforms are removed from validation, test, and
            prediction datasets (default: `None`).
        force_reload: Whether to re-process the dataset (default: `False`).
        search_kwargs: Optional dataset-specific search kwargs (used by Material Project).
        **kwargs: Additional keyword arguments to be passed to the dataset (if `dataset` is not
            used) or to the `torch_geometric.loader.DataLoader` object.
    """

    def __init__(
        self,
        *,
        dataset: Dataset | None = None,
        dataset_name: str | None = None,
        lengths: Sequence[int | float] | None = None,
        pred_dataset: Dataset | None = None,
        pre_filters: Any = None,  # noqa: ANN401
        pre_transforms: Any = None,  # noqa: ANN401
        transforms: Any = None,  # noqa: ANN401
        use_imbalance_sampler: bool = False,
        force_reload: bool = False,
        search_kwargs: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> None:
        if dataset is None and dataset_name is None and pred_dataset is None:
            raise ValueError(
                "Either `dataset`, `dataset_name`, or `pred_dataset` must be provided."
            )

        if dataset_name is not None and dataset_name not in DATASET_FACTORY:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Available datasets: {DATASET_FACTORY.keys()}"
            )

        if lengths is not None and len(lengths) not in {1, 2, 3}:
            raise ValueError(f"Invalid lengths: {lengths}. Expected 1, 2, or 3 values.")

        if kwargs.get("sampler", None) is not None and use_imbalance_sampler:
            raise ValueError("Cannot use both `sampler` and `use_imbalance_sampler`.")

        super().__init__()

        self.save_hyperparameters(
            logger=False,
            ignore=["transforms"],
        )  # see for pre_filters and pre_transforms

        kwargs.pop("shuffle", None)
        kwargs["num_workers"] = kwargs.get("num_workers", 0)
        kwargs["pin_memory"] = kwargs.get("pin_memory", True)
        kwargs["persistent_workers"] = kwargs.get("persistent_workers", kwargs["num_workers"] > 0)
        kwargs["multiprocessing_context"] = kwargs.get("multiprocessing_context", None)

        self.kwargs = kwargs

        self.dataset_name = dataset_name if dataset is None else None
        self.dataset: Dataset | None = dataset
        self.lengths = lengths

        self._batch_size = kwargs.get("batch_size", 1)
        self.kwargs["batch_size"] = self._batch_size

        self.use_imbalance_sampler = use_imbalance_sampler

        pre_filters = class_instantiator(pre_filters)
        pre_transforms = class_instantiator(pre_transforms)
        transforms = class_instantiator(transforms)

        if pre_filters is not None and isinstance(pre_filters, list):
            pre_filters = T.ComposeFilters(pre_filters)
        if isinstance(pre_transforms, list):
            pre_transforms = T.Compose(pre_transforms)
        if isinstance(transforms, list):
            transforms = T.Compose(transforms)

        log = kwargs.pop("log", False)
        download_only = kwargs.pop("download_only", False)
        k = kwargs.pop("k", 12)
        self.dataset_kwargs = {
            "transform": transforms,
            "pre_transform": pre_transforms,
            "pre_filter": pre_filters,
            "log": log,
            "force_reload": force_reload,
            "download_only": download_only,
            "k": k,
        }

        root = kwargs.pop("root", None)
        if root is not None:
            self.dataset_kwargs["root"] = root

        if search_kwargs is not None:
            self.dataset_kwargs["search_kwargs"] = dict(search_kwargs)

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        self.pred_dataset: Dataset | None = pred_dataset
        self._warned_train_only_transforms = False

    @property
    def num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        if self.dataset is None and self.dataset_name is not None:
            self.dataset = DATASET_FACTORY[self.dataset_name](**self.dataset_kwargs)

        assert self.dataset is not None
        return self.dataset.num_classes

    @property
    def batch_size(self) -> int:
        """The batch size to be used in the dataloader."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self._batch_size = value
        self.kwargs["batch_size"] = value

    def prepare_data(self) -> None:
        """Download the dataset."""
        if self.dataset_name is None or self.dataset is not None:
            return

        kwargs = copy.copy(self.dataset_kwargs)
        kwargs.pop("download_only", None)
        kwargs.pop("log", None)

        # Check if the dataset needs to be downloaded. We pass the transforms to the dataset
        # to avoid warnings when the dataset was already processed with transforms.
        DATASET_FACTORY[self.dataset_name](
            log=False,
            download_only=True,
            **self.dataset_kwargs,
        )

    def setup(self, stage: Stage) -> None:
        """Load the dataset and set the train, validation, and test datasets."""
        if stage == "predict":
            if self.pred_dataset is not None:
                self._remove_train_only_transforms(self.pred_dataset)
            return

        # Create a dataset instance only if it was not provided/created before.
        # It avoids reloading the dataset at each call to `setup`.
        if self.dataset is None and self.dataset_name is not None:
            self.dataset = DATASET_FACTORY[self.dataset_name](
                **self.dataset_kwargs,
            )

        # Make sure the dataset is split only once.
        if self.train_dataset is None:
            assert self.dataset is not None

            if self.lengths is None:
                self.train_dataset = self.dataset
                return

            split_map = {
                1: ("train_dataset",),
                2: ("train_dataset", "val_dataset"),
                3: ("train_dataset", "val_dataset", "test_dataset"),
            }
            datasets = random_split(dataset=self.dataset, lengths=self.lengths)
            for attr, dataset in zip(split_map[len(self.lengths)], datasets):
                setattr(self, attr, dataset)

            # Evaluation splits retain deterministic transforms but exclude training augmentation.
            for dataset in (self.val_dataset, self.test_dataset):
                if dataset is not None:
                    self._remove_train_only_transforms(dataset)

    def dataloader(self, dataset: Dataset, **kwargs) -> DataLoader:
        """Return a DataLoader for the given dataset."""
        kwargs.pop("k", None)

        return DataLoader(dataset, **kwargs)

    @staticmethod
    def _split_runtime_transform(transform: object) -> tuple[object | None, list[str]]:
        """Split a runtime pipeline into an evaluation-safe pipeline and removed augmentations.

        Args:
            transform: A runtime transform, a nested ``T.Compose`` pipeline, or ``None``.

        Returns:
            A tuple containing the pipeline with train-only transforms removed (or ``None``) and
            the class names of the removed train-only transforms.
        """
        # Case 1: No transform or a single train-only transform
        if transform is None or isinstance(transform, TRAIN_ONLY_TRANSFORMS):
            # A train-only transform has no evaluation-stage replacement.
            removed_transform_names = [type(transform).__name__] if transform is not None else []
            # No transform left for evaluation, but return the names of the removed transforms.
            return None, removed_transform_names

        # Case 2: A nested pipeline of transforms
        if isinstance(transform, T.Compose):
            surviving_transforms, removed_transform_names = [], []

            for transform_child in transform.transforms:
                # Recurse : deterministic transforms survive for composed pipelines of transforms
                transform_child, child_removed = LightningDataset._split_runtime_transform(
                    transform_child
                )

                if transform_child is not None:
                    surviving_transforms.append(transform_child)
                removed_transform_names.extend(child_removed)

            if not surviving_transforms:
                return None, removed_transform_names

            # Avoid wrapping a single surviving transform in an unnecessary Compose.
            evaluation_transform = (
                surviving_transforms[0]
                if len(surviving_transforms) == 1
                else T.Compose(surviving_transforms)
            )
            return evaluation_transform, removed_transform_names

        # Case 3: A single deterministic transform
        return transform, []

    def _remove_train_only_transforms(self, dataset: Dataset) -> None:
        """Remove train-only runtime transforms from an evaluation dataset.

        Args:
            dataset: Validation, test, or prediction dataset whose runtime pipeline is filtered.

        Returns:
            None. The dataset's ``transform`` is updated in place only when train-only transforms
            are present.
        """
        transform: object = dataset.transform
        evaluation_transform, removed = self._split_runtime_transform(transform)

        # No train-only transforms were found, so no update is needed.
        if not removed:
            return

        # Keep deterministic transforms while preventing stochastic augmentation at evaluation.
        dataset.transform = evaluation_transform  # type: ignore[assignment]
        if not self._warned_train_only_transforms:
            # Emit one warning even when validation and test each require filtering.
            warnings.warn(
                "Train-only transform(s) will be ignored for validation, test, and prediction: "
                f"{', '.join(removed)}",
                stacklevel=2,
            )
            self._warned_train_only_transforms = True  # No need to warn again

    def train_dataloader(self) -> DataLoader:
        """Return a DataLoader for the training dataset. The dataset is shuffled if it is not an
        iterable dataset and no sampling technique is used.
        """
        from torch.utils.data import IterableDataset

        if self.train_dataset is None:
            self.setup("fit")
        assert self.train_dataset is not None

        kwargs = copy.copy(self.kwargs)

        # Workaround to use do graph-level sampling with node-level labels
        if self.use_imbalance_sampler:
            sampler = ImbalancedSampler(
                torch.tensor([data.y[0].item() for data in self.train_dataset])
            )
            kwargs["sampler"] = sampler

        shuffle = not isinstance(self.train_dataset, IterableDataset)
        shuffle &= kwargs.get("sampler", None) is None
        shuffle &= kwargs.get("batch_sampler", None) is None

        return self.dataloader(self.train_dataset, shuffle=shuffle, **kwargs)

    def val_dataloader(self) -> DataLoader:
        """Return a DataLoader for the validation dataset. The dataset is not shuffled and no
        sampling technique is used.
        """
        assert self.val_dataset is not None

        kwargs = copy.copy(self.kwargs)
        kwargs.pop("sampler", None)
        kwargs.pop("batch_sampler", None)

        return self.dataloader(self.val_dataset, shuffle=False, **kwargs)

    def test_dataloader(self) -> DataLoader:
        """Return a DataLoader for the test dataset. The dataset is not shuffled and no
        sampling technique is used.
        """
        assert self.test_dataset is not None

        kwargs = copy.copy(self.kwargs)
        kwargs.pop("sampler", None)
        kwargs.pop("batch_sampler", None)

        return self.dataloader(self.test_dataset, shuffle=False, **kwargs)

    def predict_dataloader(self) -> DataLoader:
        """Return a DataLoader for the test dataset. The dataset is not shuffled and no
        sampling technique is used.
        """
        assert self.pred_dataset is not None
        self._remove_train_only_transforms(self.pred_dataset)

        kwargs = copy.copy(self.kwargs)
        kwargs.pop("sampler", None)
        kwargs.pop("batch_sampler", None)

        return self.dataloader(self.pred_dataset, shuffle=False, **kwargs)

    def __repr__(self) -> str:
        kwargs = kwargs_repr(
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            test_dataset=self.test_dataset,
            pred_dataset=self.pred_dataset,
            **self.kwargs,
        )
        return f"{self.__class__.__name__}({kwargs})"


def kwargs_repr(**kwargs) -> str:
    """Return a string representation of the keyword arguments."""
    return ", ".join([f"{k}={v}" for k, v in kwargs.items() if v is not None])
