import copy
import warnings
from typing import Any

from lightning import LightningDataModule as PLLightningDataModule
from torch.utils.data import DataLoader, Dataset


class LightningDataModule(PLLightningDataModule):
    def __init__(self, has_val: bool, has_test: bool, **kwargs: Any) -> None:
        super().__init__()

        if not has_val:
            self.val_dataloader = None  # type: ignore

        if not has_test:
            self.test_dataloader = None  # type: ignore

        kwargs.setdefault("batch_size", 1)
        kwargs.setdefault("num_workers", 0)
        kwargs.setdefault("pin_memory", True)
        kwargs.setdefault("persistent_workers", kwargs.get("num_workers", 0) > 0)

        if "shuffle" in kwargs:
            warnings.warn(
                f"The 'shuffle={kwargs['shuffle']}' option is "
                f"ignored in '{self.__class__.__name__}'. Remove it "
                f"from the argument list to disable this warning"
            )
            del kwargs["shuffle"]

        self.kwargs = kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({kwargs_repr(**self.kwargs)})"


class LightningDataset(LightningDataModule):
    r"""Converts a set of `torch_geometric.data.Dataset` objects into a
    `pytorch_lightning.LightningDataModule` variant. It can then be
    automatically used as a `datamodule` for multi-GPU graph-level
    training via
    `PyTorch Lightning <https://www.pytorchlightning.ai>`__.
    `LightningDataset` will take care of providing mini-batches via
    `torch_geometric.loader.DataLoader`.

        .. code-block::

            import pytorch_lightning as pl
            trainer = pl.Trainer(strategy="ddp_spawn", accelerator="gpu",
                                 devices=4)
            trainer.fit(model, datamodule)

    Args:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset, optional): The validation dataset.
            (default: `None`)
        test_dataset (Dataset, optional): The test dataset.
            (default: `None`)
        pred_dataset (Dataset, optional): The prediction dataset.
            (default: `None`)
        **kwargs (optional): Additional arguments of
            `torch_geometric.loader.DataLoader`.
    """

    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
        pred_dataset: Dataset | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            has_val=val_dataset is not None,
            has_test=test_dataset is not None,
            **kwargs,
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.pred_dataset = pred_dataset

    def dataloader(self, dataset: Dataset, **kwargs: Any) -> DataLoader:
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self) -> DataLoader:
        from torch.utils.data import IterableDataset

        shuffle = not isinstance(self.train_dataset, IterableDataset)
        shuffle &= self.kwargs.get("sampler", None) is None
        shuffle &= self.kwargs.get("batch_sampler", None) is None

        return self.dataloader(
            self.train_dataset,
            shuffle=shuffle,
            **self.kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None

        kwargs = copy.copy(self.kwargs)
        kwargs.pop("sampler", None)
        kwargs.pop("batch_sampler", None)

        return self.dataloader(self.val_dataset, shuffle=False, **kwargs)

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None

        kwargs = copy.copy(self.kwargs)
        kwargs.pop("sampler", None)
        kwargs.pop("batch_sampler", None)

        return self.dataloader(self.test_dataset, shuffle=False, **kwargs)

    def predict_dataloader(self) -> DataLoader:
        assert self.pred_dataset is not None

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


def kwargs_repr(**kwargs: Any) -> str:
    return ", ".join([f"{k}={v}" for k, v in kwargs.items() if v is not None])
