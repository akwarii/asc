from collections.abc import Sequence
from typing import Any, Literal

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from src.data import _REPR_INDENT, datasets
from src.data import transforms as T
from src.utils.typing import PathLike, StageType

DATASET_MAPPING = {
    "aflow": datasets.Aflow,
    "mp": datasets.MaterialProject,
    "gnome": datasets.Gnome,
}


# TODO add collate_fn to dataloader
# TODO add neighbor sampler
class CEGANNDataModule(LightningDataModule):
    """CEGANNDataModule is a LightningDataModule subclass that provides data loading and processing
    functionality for the CEGANN model.

    Args:
        root (str, optional): The root directory where the datasets are stored. Defaults to "data".
        datasets (Sequence[Literal["aflow", "mp", "gnome"]], optional): The datasets to use.
            Multiple datasets can be used at once. Defaults to ("gnome",).
        train_val_test_split (tuple[int, int, int] | tuple[float, float, float], optional): The split ratios for train,
            validation, and test datasets. If integers are used, the split ratios are calculated as the number of samples
            for each dataset. If floats are used, the split ratios are calculated as the percentage of samples for each
            dataset (). Defaults to (0.8, 0.1, 0.1).
        transforms (Any, optional): The data transformations (eg. normalization) to apply. Transformations are applied to
            the structure graph. Defaults to None.
        struct_transforms (Any, optional): The structure transformations (eg. random noise) to apply. Structure
            transformations are applied to the structure (before the graph creation). Defaults to None.
        batch_size (int, optional): The batch size for data loading. Defaults to 64.
        num_workers (int, optional): The number of workers for data loading. Defaults to 0.
        pin_memory (bool, optional): Whether to pin memory for faster data transfer. Defaults to False.
        **kwargs: Additional keyword arguments.

    Methods:
        prepare_data: Safely download and save the dataset.
        setup: Load data in memory.
        train_dataloader: Create and return the train dataloader.
        val_dataloader: Create and return the validation dataloader.
        test_dataloader: Create and return the test dataloader.
        predict_dataloader: Create and return the predict dataloader.
        teardown: Cleans up the data after a specific stage.
        transfer_batch_to_device: Override this hook if your DataLoader returns tensors wrapped in a custom data
            structure.
        on_before_batch_transfer: Override to alter or apply batch augmentations to your batch before it is transferred to
            the device.
        on_after_batch_transfer: Override to alter or apply batch augmentations to your batch after it is transferred to
            the device.
        state_dict: Called when saving a checkpoint. Implement to generate and save the datamodule state.
        load_state_dict: Called when loading a checkpoint. Implement to reload datamodule state given datamodule
            `state_dict()`.
        __repr__: Return a string representation of the datamodule.
    """

    _repr_indent = _REPR_INDENT

    def __init__(
        self,
        root: PathLike = "data",
        datasets: Sequence[Literal["aflow", "mp", "gnome"]] = ("gnome",),
        train_val_test_split: tuple[int, int, int] | tuple[float, float, float] = (0.8, 0.1, 0.1),
        transforms: Sequence[Any] | None = None,
        struct_transforms: Sequence[Any] | None = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        if transforms is None:
            self.transforms = None
        else:
            self.transforms = T.Compose([t for t in transforms])

        if struct_transforms is None:
            self.struct_transforms = None
        else:
            self.struct_transforms = T.Compose([t for t in struct_transforms])

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None
        self.data_predict: Dataset | None = None

    @property
    def num_classes(self) -> int | None:
        """Get the number of classes.

        If the dataset is not loaded, return None.
        """
        # Once setup has been called, either `self.data_train` or `self.data_predict` will be set
        if self.data_train:
            return len(self.data_train.classes)
        if self.data_predict:
            return len(self.data_predict.classes)

    def prepare_data(self) -> None:
        """Safely download and save the dataset.

        This method is called only from a single process.
        In case of multi-node training, the execution of this hook depends upon `prepare_data_per_node`.
        """
        for dataset in self.hparams.datasets:
            DATASET_MAPPING[dataset](self.hparams.root, download=True, load=False)

    def setup(self, stage: StageType) -> None:
        """Load data in memory. If stage is either `"fit"`, `"validate"` or `"test"`, then
        `self.data_train`, `self.data_val` and `self.data_test` will be set. If stage is
        `"predict"` then `self.data_predict` will be set.

        The random split is done only once and the same split is used for all the stages (except for the `"predict"` stage which use the
        full prediction dataset). This is because the random split is deterministic and the same seed is used for all the stages.

        Args:
            stage: The stage to load the data for. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        # We only test for self.data_test because if self.data_train is set,
        # then self.data_val and self.data_test are also set
        if stage != "predict" and not self.data_test:
            dataset = ConcatDataset(
                [
                    DATASET_MAPPING[dataset](
                        self.hparams.root,
                        transform=self.transforms,
                        struct_transform=self.struct_transforms,
                    )
                    for dataset in self.hparams.datasets
                ]
            )
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

        if stage == "predict":
            dataset = ConcatDataset(
                [
                    DATASET_MAPPING[dataset](
                        self.hparams.root,
                        transform=self.transforms,
                        struct_transform=self.struct_transforms,
                    )
                    for dataset in self.hparams.datasets
                ]
            )
            self.data_predict = dataset

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        Returns:
            The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        Returns:
            The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        Returns:
            The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the predict dataloader.

        Returns:
            The predict dataloader.
        """
        return DataLoader(
            dataset=self.data_predict,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: StageType) -> None:
        """Cleans up the data after a specific stage. Note that the test dataloader is the last to
        be used before performing predictions. Meaning that if the stage is `"test"`, then
        `self.data_train`, `self.data_val` and `self.data_test` will be set to `None`. If the stage
        is `"predict"`, then `self.data_predict` will be set to `None`.

        Args:
            stage: The stage to clean up the data for.
        """
        # !Assume that test dataloader is the last to be used before performing predictions
        if stage == "test":
            self.data_train = None
            self.data_val = None
            self.data_test = None

        if stage == "predict":
            self.data_predict = None

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        """Override this hook if your DataLoader returns tensors wrapped in a custom data
        structure. The data types listed below (and any arbitrary nesting of them) are supported
        out of the box: `torch.Tensor` or anything that implements .to(…) `list` `dict` `tuple` For
        anything else, you need to define how the data is moved to the target device (CPU, GPU,
        TPU, …).

        Args:
            batch: The output of the DataLoader.
            device: The target device.
            dataloader_idx: The index of the DataLoader producing the batch.

        Returns:
            The batch after applying the batch augmentations.
        """
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """Override to alter or apply batch augmentations to your batch before it is transferred to
        the device.

        Args:
            batch: The output of the DataLoader.
            dataloader_idx: The index of the DataLoader producing the batch.

        Returns:
            The batch after applying the batch augmentations.
        """
        return super().on_before_batch_transfer(batch, dataloader_idx)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """Override to alter or apply batch augmentations to your batch after it is transferred to
        the device.

        Args:
            batch: The output of the DataLoader.
            dataloader_idx: The index of the DataLoader producing the batch.

        Returns:
            The batch after applying the batch augmentations.
        """
        return super().on_after_batch_transfer(batch, dataloader_idx)

    def state_dict(self) -> dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        Returns:
            A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        Args:
            state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

    def __repr__(self) -> str:
        format_string = f"{self.__class__.__name__}(\n"
        for k, v in self.hparams.items():
            if v is not None:
                format_string += " " * self._repr_indent + f"{k}={v}\n"


def kwargs_repr(**kwargs: Any) -> str:
    return ", ".join([f"{k}={v}" for k, v in kwargs.items() if v is not None])


if __name__ == "__main__":
    CEGANNDataModule()
