from collections.abc import Sequence
from typing import Any

import torch
import torch_geometric.transforms as T
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

import src.data.datasets as datasets
from src.processing.graph import KNNGraph  # DB
from src.utils.constants import REPR_INDENT
from src.utils.typing import PathLike, StageType

DATASET_MAP = {
    "aflow": datasets.Aflow,
    "mp": datasets.MaterialProject,
    "gnome": datasets.Gnome,
    "csg": datasets.CSG,
    "custom": datasets.CustomDataset,
}


# These lines are (sadly) needed to avoid pyright errors due to the use of self.hparams
# pyright: reportAttributeAccessIssue=false
# pyright: reportAssignmentType=false
# pyright: reportArgumentType=false
# pyright: reportOptionalIterable=false
# TODO add collate_fn to dataloader ~~~~ DONE ?
# TODO integrate dynamic batch and imbalanced sampling
# TODO integrate node loader
class CEGANNDataModule(LightningDataModule):
    """CEGANNDataModule is a LightningDataModule subclass that provides data loading and processing
    functionality for the CEGANN model.

    Args:
        root (str, optional): The root directory where the datasets are stored. Defaults to "data".
        datasets (Sequence[str] | str, optional): The datasets to use.
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
        seed (int, optional): The seed to use for the random split. Defaults to 42.
        k_neigh (int, optional): Number of neighbors to use in the K-nearest Neighbors graphs.
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

    def __init__(
        self,
        root: PathLike = "data",
        datasets: Sequence[str] | str = "csg",
        train_val_test_split: tuple[int, int, int] | tuple[float, float, float] = (0.8, 0.1, 0.1),
        transforms: Sequence[Any] | None = None,
        struct_transforms: Sequence[Any] | None = None,
        aumgenter_transforms: Sequence[Any] | None = None,
        sampler: Any | None = None,
        batch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
        k_neigh: int = None,  # DB
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(datasets, str):
            datasets = [datasets]

        if any(dataset not in DATASET_MAP for dataset in datasets):
            raise ValueError(f"Invalid dataset. Available datasets are {list(DATASET_MAP.keys())}")

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # DB - Number of neighbors
        self.k_neigh = k_neigh

        # DB - other arguments
        self.kwargs = kwargs

        # data transformations
        if transforms is None:
            self.transforms = None
        else:
            self.transforms = T.Compose([t for t in transforms])

        if struct_transforms is None:
            self.struct_transforms = None
        else:
            self.struct_transforms = T.Compose([t for t in struct_transforms])

        if aumgenter_transforms is None:
            self.augmenter_transforms = None
        else:
            self.augmenter_transforms = T.Compose([t for t in aumgenter_transforms])

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None
        self.data_predict: Dataset | None = None

    @property
    def num_classes(self) -> int | None:
        """Get the number of classes.

        If the dataset is not loaded, return None.
        """
        # TODO: Change the function as self.data_xxxx are Subsets, not Datasets and have different attributes
        # Once setup has been called, either `self.data_train` or `self.data_predict` will be set
        if self.data_train:
            return len(self.data_train.classes)
        elif self.data_predict:
            return len(self.data_predict.classes)
        else:
            return None

    def prepare_data(self) -> None:
        """Safely download and save the dataset.

        This method is called only from a single process.
        In case of multi-node training, the execution of this hook depends upon `prepare_data_per_node`.
        """
        for dataset in self.hparams.datasets:
            DATASET_MAP[dataset](self.hparams.root, fetch_data=True)

    def setup(self, stage: StageType) -> None:
        """Load data in memory. If stage is either `"fit"`, `"validate"` or `"test"`, then
        `self.data_train`, `self.data_val` and `self.data_test` will be set. If stage is
        `"predict"` then `self.data_predict` will be set.

        The random split is done only once and the same split is used for all the stages (except for the `"predict"` stage which use the
        full prediction dataset). This is because the random split is deterministic and the same seed is used for all the stages.

        Args:
            stage: The stage to load the data for. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        # DB - Number of neighbors management
        graph_kwargs = {}
        if self.k_neigh:
            graph_kwargs["k"] = self.k_neigh
        kwargs, keys_tokeep = {}, ["pretreat", "origin_dir"]
        if self.kwargs:
            kwargs = {key: self.kwargs[key] for key in keys_tokeep}
        # We only test for self.data_test because if self.data_train is set,
        # then self.data_val and self.data_test are also set
        if stage != "predict" and not self.data_test:
            dataset = ConcatDataset(
                [
                    DATASET_MAP[dataset](
                        self.hparams.root,
                        transform=self.transforms,
                        struct_transform=self.struct_transforms,
                        graph_kwargs=graph_kwargs,
                        **kwargs,
                    )
                    for dataset in self.hparams.datasets
                ]
            )
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(self.hparams.seed),
            )

        if stage == "predict":
            dataset = ConcatDataset(
                [
                    DATASET_MAP[dataset](
                        self.hparams.root,
                        transform=self.transforms,
                        struct_transform=self.struct_transforms,
                        graph_kwargs=graph_kwargs,
                        **kwargs,
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
            collate_fn=KNNGraph.collate,  # DB
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
            collate_fn=KNNGraph.collate,  # DB
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
            collate_fn=KNNGraph.collate,  # DB
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
            collate_fn=KNNGraph.collate,  # DB
        )

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """Apply batch augmentations to the batch before it is transferred to the device. Both the
        structure and the data transformations are applied (in this order).

        Args:
            batch: The batch to augment.
            dataloader_idx: The index of the dataloader.
        """
        if self.trainer.training:  # type: ignore
            if self.struct_transforms is not None:
                batch = self.struct_transforms(batch)
            if self.transforms is not None:
                batch = self.transforms(batch)
            if self.augmenter_transforms is not None:
                batch = self.augmenter_transforms(batch)
        return batch

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

    def __repr__(self) -> str:
        format_string = f"{self.__class__.__name__}(\n"
        for k, v in self.hparams.items():
            if v is not None:
                format_string += " " * REPR_INDENT + f"{k}={v}\n"
        return format_string + ")"
