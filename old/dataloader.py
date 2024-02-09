import logging
import sys
from collections.abc import Callable
from typing import Optional

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

logging.getLogger(__name__)


def get_train_val_test_loader(
    dataset: Dataset,
    collate_fn: Callable = default_collate,
    batch_size: int = 64,
    train_ratio: float | None = None,
    val_ratio: float | None = 0.1,
    test_ratio: float | None = 0.1,
    num_workers: int | None = 1,
    pin_memory: bool = False,
    **kwargs,
):
    """Returns train, validation, and test data loaders for a given dataset.

    Args:
        dataset (Dataset): The dataset to be used for creating the data loaders.
        collate_fn (callable, optional): Function used to collate the samples into a batch. Defaults to default_collate.
        batch_size (int, optional): Number of samples per batch. Defaults to 64.
        train_ratio (float, optional): Ratio of training data. If not provided, it is calculated as 1 - val_ratio - test_ratio. Defaults to None.
        val_ratio (float, optional): Ratio of validation data. Defaults to 0.1.
        test_ratio (float, optional): Ratio of test data. Defaults to 0.1.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 1.
        pin_memory (bool, optional): If True, the data loader will copy tensors into pinned memory. Defaults to False.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple: A tuple containing the train, validation, and test data loaders.
    """
    total_size = dataset.size

    if kwargs["train_size"] is None:
        if train_ratio is None:
            train_ratio = 1 - val_ratio - test_ratio
            logging.warning(
                f"train_ratio is not set, defaulting to 1-val_ratio-test_ratio = {train_ratio}."
            )
        if train_ratio + val_ratio + test_ratio > 1:
            logging.error("The sum of train_ratio, val_ratio, test_ratio is greater than 1.")
            sys.exit()

    indices = list(range(total_size))
    if kwargs["train_size"] is not None:
        train_size = kwargs["train_size"]
    else:
        train_size = int(train_ratio * total_size)
    if kwargs["test_size"] is not None:
        test_size = kwargs["test_size"]
    else:
        test_size = int(test_ratio * total_size)
    if kwargs["val_size"] is not None:
        valid_size = kwargs["val_size"]
    else:
        valid_size = int(val_ratio * total_size)

    train_sampler = SubsetRandomSampler(indices[:train_size])

    if test_size == 0:
        val_sampler = SubsetRandomSampler(indices[-valid_size:])
    else:
        val_sampler = SubsetRandomSampler(indices[-(valid_size + test_size) : -test_size])
        test_sampler = SubsetRandomSampler(indices[-test_size:])

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    if test_size == 0:
        test_loader = []
    else:
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader, test_loader
