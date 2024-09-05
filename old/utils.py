from __future__ import annotations

import logging
import multiprocessing
import shutil
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from graph import CrystalGraphDataset
from pymatgen.core.structure import IStructure
from pymatgen.io.vasp.inputs import Poscar
from settings import Settings
from tqdm import tqdm
from yaml import full_load

logging.getLogger(__name__)


def prepare_batch_fn(batch: tuple[torch.TensorType, ...], device: str, non_blocking: bool = False):
    """Helper function to move the tensors to the device. The tensors can be moved asynchronously
    by setting non_blocking to True.

    Args:
        batch (tuple): A tuple containing the following elements:
            - bond_feature (torch.Tensor): Tensor representing bond features.
            - angular_feature (torch.Tensor): Tensor representing angular features.
            - neighbor_idx (torch.Tensor): Tensor representing neighbor indices.
            - crystal_idx (torch.Tensor): Tensor representing crystal indices.
            - target (torch.Tensor): Tensor representing the target values.

        device (torch.device): The device to move the tensors to.
        non_blocking (bool): If True, the tensors are moved to the device asynchronously.

    Returns:
        tuple: A tuple containing the following elements:
            - bond_feature (torch.Tensor): Tensor representing bond features on the specified device.
            - angular_feature (torch.Tensor): Tensor representing angular features on the specified device.
            - neighbor_idx (torch.Tensor): Tensor representing neighbor indices on the specified device.
            - crystal_idx (torch.Tensor): Tensor representing crystal indices on the specified device.

        torch.Tensor: Tensor representing the target values on the specified device.
    """
    (bond_feature, angular_feature, neighbor_idx, crystal_idx, target) = batch

    return (
        bond_feature.to(device, non_blocking=non_blocking),
        angular_feature.to(device, non_blocking=non_blocking),
        neighbor_idx.to(device, non_blocking=non_blocking),
        crystal_idx.to(device, non_blocking=non_blocking),
    ), target.to(device, non_blocking=non_blocking)


def load_settings(config_file: str | Path | None = "custom_config.yaml") -> Settings:
    """Load the settings from a custom configuration file.

    Args:
        config_file (Optional[str | Path]): The path to the custom configuration file.
            If not provided, the default file "custom_config.yaml" will be used.

    Returns:
        Settings: The loaded settings.
    """
    if not isinstance(config_file, Path):
        config_file = Path(config_file)

    if config_file.exists():
        with open(config_file) as file:
            custom_dict = full_load(file)
            settings = Settings(**custom_dict)
    else:
        print("No custom configs found, using default settings")
        logging.warning("No custom configs found, using default settings")
        settings = Settings()

    return settings


def process_poscar(
    poscar_path: str | Path, allow_unknown: bool
) -> dict[str, np.ndarray | IStructure]:
    poscar = Poscar.from_file(poscar_path)

    try:
        target = [label for label in poscar.comment.split(",")]
    except ValueError:
        if allow_unknown:
            target = [1]
        else:
            raise ValueError("Target values must be provided in the POSCAR comment")

    data = {
        "structure": poscar.structure,
        "target": np.array(target, dtype=np.short),
    }

    return data


def load_dataset(
    data_dir: str | Path,
    settings: Settings,
    return_labels: bool | None = False,
    allow_unknown: bool | None = False,
) -> CrystalGraphDataset | tuple[CrystalGraphDataset, list[str]]:
    """Load a dataset from the specified directory.

    Args:
        data_dir (str | Path): The directory path where the dataset is located.
        settings (Settings): The settings object containing configuration parameters.
        return_labels (Optional[bool], optional): Whether to return the labels along with the dataset. Defaults to False.
        allow_unknown (Optional[bool], optional): Whether to allow unknown target values. Defaults to False.

    Returns:
        CrystalGraphDataset | tuple[CrystalGraphDataset, list[str]]: The loaded dataset. If return_labels is True, it returns a tuple containing the dataset and the list of labels.

    Raises:
        ValueError: If target values are not provided in the POSCAR comment and allow_unknown is False.
    """
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)

    poscars = sorted([f.as_posix() for f in data_dir.glob("*.POSCAR")])

    if return_labels:
        labels = [Path(poscar_path).stem for poscar_path in poscars]

    with multiprocessing.Pool(settings.num_workers) as pool:
        func = partial(process_poscar, allow_unknown=allow_unknown)
        classification_dataset = list(
            tqdm(pool.imap_unordered(func, poscars), total=len(poscars), desc="Loading dataset")
        )

    graphs = CrystalGraphDataset(
        classification_dataset,
        neighbors=settings.neighbors,
        rcut=settings.rcut,
        delta=settings.search_delta,
        mp_load=True if settings.num_workers > 1 else False,
        mp_cpu_count=settings.num_workers,
    )

    if return_labels:
        return graphs, labels
    else:
        return graphs


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    best_val_accuracy: float,
    epoch: int,
    is_best: bool,
    path: str | Path | None = Path.cwd(),
    filename: str | Path | None = "checkpoint.pt",
) -> None:
    """Save the checkpoint of the training process.

    Args:
        model (nn.Module): The model being trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler used for training.
        best_val_accuracy (float): The best validation accuracy achieved during training.
        epoch (int): The current epoch number.
        is_best (bool): Whether the current checkpoint is the best one.
        path (str | Path, optional): The directory path to save the checkpoint. Defaults to current working directory.
        filename (str | Path, optional): The filename of the checkpoint. Defaults to "checkpoint.pt".

    Returns:
        None
    """
    if not isinstance(path, Path):
        path = Path(path)

    fname = path / f"{filename}_{epoch}"
    torch.save(
        {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "best_accuracy": best_val_accuracy,
        },
        fname,
    )

    if is_best:
        shutil.copyfile(fname, path / "model_best.pt")


def resume_training(output_dir, model, optimizer, scheduler, trainer, settings):
    """Resumes training from a checkpoint file.

    Args:
        output_dir (str): The directory where the checkpoint file is located.
        model: The model to load the state dictionary into.
        optimizer: The optimizer to load the state dictionary into.
        scheduler: The scheduler to load the state dictionary into.
        trainer: The trainer to load the state dictionary into.
        settings: The settings object containing the number of epochs.

    Raises:
        FileNotFoundError: If the checkpoint file is not found.
    """
    if Path(f"{output_dir}/checkpoint.pt").is_file():
        logging.info(f"Loaded checkpoint: '{output_dir}/checkpoint.pt'")
        ckpt = torch.load(f"{output_dir}/checkpoint.pt")

        epoch = ckpt["epoch"]
        settings.epochs = settings.epochs - epoch - 1

        best_val_error = ckpt["best_accuracy"]

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["lr_scheduler"])
        trainer.load_state_dict(ckpt["trainer"])
    else:
        raise FileNotFoundError("Checkpoint not found")
