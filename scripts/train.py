import argparse
from pathlib import Path

import torch
import torchmetrics
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateFinder,
    ModelCheckpoint,
)
from src.callbacks import HalfBatchSizeFinder
from src.constants import DEFAULT_SEED
from src.datamodule import LightningDataset
from src.module import Module
from src.transforms import LineGraph, RandomPerturbation
from src.utils.cli import KeyValueParserAction


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate the parsed arguments."""
    if args.use_best_model and not args.storage:
        parser.error("--use_best_model requires the --storage argument to be set.")
    if args.use_best_model and args.model_kwargs:
        parser.error("--use_best_model cannot be used with --model_kwargs.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training loop")
    parser.add_argument(
        "--model",
        type=str,
        choices=("cegann", "mlp", "gat"),
        default="cegann",
        help="Model to optimize the hyperparameters for.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size to use for the training.",
    )
    parser.add_argument(
        "--batch_size_finder",
        action="store_true",
        help="Find the optimal batch size. Overrides the batch_size argument.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of epochs to train the model for.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("csg", "mp", "aflow", "gnome", "custom"),
        default="csg",
        help="Name of the dataset to use for the training.",
    )
    parser.add_argument(
        "--model_kwargs",
        action=KeyValueParserAction,
        help="Model hyperparameters as key=value pairs.",
    )
    parser.add_argument(
        "--use_best_model",
        action="store_true",
        help=(
            "Use the best architecture found for the model during HPO. "
            "--study argument is required."
        ),
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///hpo.db",
        help="URL to the database to store the optimization results.",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable model compilation.",
    )

    args = parser.parse_args()
    validate_args(args, parser)

    return args


def get_best_model_params(storage, study) -> dict:
    try:
        import optuna
    except ImportError:
        raise ImportError("Optuna package is required to load the best model.")

    storage_path = Path(storage.split(":///")[-1])
    if not storage_path.exists():
        raise FileNotFoundError(f"Storage '{storage}' not found.")

    try:
        study = optuna.load_study(study_name=study, storage=storage)
    except KeyError:
        raise ValueError(f"Study '{study}' not found in the storage '{storage}'.")

    return study.best_params


def main() -> None:
    seed_everything(DEFAULT_SEED)

    args = parse_args()

    if args.use_best_model:
        best_params = get_best_model_params(args.storage, args.model)
        args.model_kwargs = best_params

    if args.model_kwargs.get("k") is None:
        raise ValueError("The number of neighbors 'k' is required as a model kwarg.")

    callbacks = [
        LearningRateFinder(min_lr=1e-5, max_lr=0.1),
        EarlyStopping(monitor="val/loss", patience=10, check_on_train_epoch_end=False),
        ModelCheckpoint(monitor="val/loss", save_on_train_epoch_end=False),
    ]
    if args.batch_size_finder:
        callbacks.insert(0, HalfBatchSizeFinder(steps_per_trial=100, init_val=64))

    trainer = Trainer(
        max_epochs=args.epochs,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=callbacks,
        deterministic=True,
    )

    datamodule = LightningDataset(
        dataset_name=args.dataset,
        lengths=(0.7, 0.2, 0.1),
        use_imbalance_sampler=True,
        pre_transforms=LineGraph(),
        transforms=RandomPerturbation(std=0.1),
        num_workers=8,
        batch_size=args.batch_size,
        k=args.model_kwargs["k"],
    )

    num_classes = datamodule.num_classes
    metrics = {
        "f1": torchmetrics.F1Score(task="multiclass", num_classes=num_classes),
    }

    with trainer.init_module():
        model = Module(
            model_name=args.model,
            num_classes=num_classes,
            compile=not args.no_compile,
            metrics=torchmetrics.MetricCollection(metrics),
            warmup=args.epochs // 100 * len(datamodule.train_dataloader()),
            max_iters=args.epochs * len(datamodule.train_dataloader()),
            model_kwargs=args.model_kwargs,
        )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.validate(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
