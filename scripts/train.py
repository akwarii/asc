import argparse
from pathlib import Path

import torch
import torchmetrics
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from src import LightningDataset, Module
from src.callbacks import HalfBatchSizeFinder
from src.constants import DEFAULT_SEED
from src.transforms import LineGraph, RandomPerturbation
from src.transforms.line_graph import LineGraphData
from src.utils.cli import KeyValueParserAction

torch.serialization.add_safe_globals([LineGraphData])


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
        choices=("cegann", "cegannv2", "mlp", "gat"),
        required=True,
        help="Model to optimize the hyperparameters for.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size to use for the training. Default: 128.",
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
        help="Number of epochs to train the model for. Default: 300.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("csg", "mp", "aflow", "gnome", "custom"),
        default="csg",
        help="Name of the dataset to use for the training. Default: csg.",
    )
    parser.add_argument(
        "--model_kwargs",
        action=KeyValueParserAction,
        nargs="+",
        help="Model hyperparameters as key=value pairs.",
    )
    parser.add_argument(
        "--use_best_model",
        action="store_true",
        help=(
            "Use the best architecture found for the model during HPO. "
            "The --storage argument must also be set. Overrides the --model_kwargs argument."
        ),
    )
    parser.add_argument(
        "--storage",
        type=str,
        help="URL to the database to store the optimization results.",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable model compilation.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        help="Path to a checkpoint to resume training from. Default: None.",
        default=None,
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
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    seed_everything(DEFAULT_SEED)

    args = parse_args()

    if args.use_best_model:
        best_params = get_best_model_params(args.storage, args.model)
        args.model_kwargs = best_params

    print(f"Training model {args.model} with hyperparameters: {args.model_kwargs}")

    if args.model_kwargs is None:
        args.model_kwargs = dict()
        if args.resume_from is not None:
            ckpt = torch.load(args.resume_from)
            args.model_kwargs = ckpt["hyper_parameters"]["model_kwargs"]
            args.model_kwargs["k"] = ckpt["datamodule_hyper_parameters"]["k"]

    if args.model_kwargs["k"] is None:
        raise ValueError(
            "Please provide the number of neighbors with the --model_kwargs argument."
        )

    callbacks = [
        ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=3),
    ]
    if args.batch_size_finder:
        callbacks.insert(0, HalfBatchSizeFinder(steps_per_trial=100, init_val=64, max_trials=6))

    trainer = Trainer(
        max_epochs=args.epochs,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        callbacks=callbacks,
        deterministic=False,
        enable_progress_bar=True,
    )

    datamodule = LightningDataset(
        dataset_name=args.dataset,
        lengths=(0.7, 0.2, 0.1),
        use_imbalance_sampler=True,
        pre_transforms=LineGraph(),
        transforms=RandomPerturbation(std=0.1),
        num_workers=8,
        batch_size=args.batch_size,
        k=args.model_kwargs.get("k"),
    )

    num_classes = datamodule.num_classes
    metrics = {
        "f1": torchmetrics.F1Score(task="multiclass", num_classes=num_classes),
        "auroc": torchmetrics.AUROC(task="multiclass", num_classes=num_classes),
        "acc": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
        "confmat": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes),
    }

    with trainer.init_module():
        model = Module(
            model_name=args.model,
            num_classes=num_classes,
            compile=not args.no_compile,
            metrics=torchmetrics.MetricCollection(metrics),
            warmup=700,
            lr=6e-3,
            max_iters=args.epochs * len(datamodule.train_dataloader()),
            model_kwargs=args.model_kwargs,
        )

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=args.resume_from)
    trainer.validate(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
