import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torchmetrics
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, RichModelSummary
from line_profiler import profile
from ovito.data import DataCollection
from ovito.io import export_file, import_file
from src import LightningDataset, Module
from src.constants import DEFAULT_SEED
from src.graph import PeriodicKNN
from src.transforms import RandomPerturbation
from src.typing import PathLike
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from tqdm.auto import tqdm

torch.serialization.add_safe_globals([RandomPerturbation])

EPOCHS = 30
NUM_NEIGHBORS = 20
COMPILE = True
CKPT_NAME = Path(".") / "lightning_logs" / "version_0" / "checkpoints" / "epoch=22-step=782.ckpt"
# CKPT_NAME = None
TO_PREDICT: list[PathLike] = [
    # Path.home() / "THESE" / "TEST" / "Si_mixture_polycrystal" / "final.cfg",
]


def train(trainer: Trainer, model: Module, datamodule: LightningDataset) -> None:
    trainer.fit(model=model, datamodule=datamodule)
    trainer.validate(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)

    del trainer, datamodule


def train_epoch(model: Module, dataloader: Iterable[Data]) -> None:
    import torch.nn.functional as F

    device = next(model.parameters()).device
    print(f"Training on device: {device}")

    opts, schs = model.configure_optimizers()

    if not isinstance(opts, list):
        opts = [opts]
    if schs is not None and not isinstance(schs, list):
        schs = [schs]

    for data in tqdm(dataloader, unit="batch"):
        data = data.to(device, non_blocking=True)

        preds: Tensor = model(data.x, data.edge_index, data.edge_attr)
        loss = F.cross_entropy(preds, torch.as_tensor(data.y))

        for opt in opts:
            opt.zero_grad(set_to_none=True)

        loss.backward()

        model._optimization_step(opts, schs)


@torch.inference_mode()
@profile
def inference(model: Module, atoms_list: Iterable[DataCollection]) -> list[Tensor]:
    knn = PeriodicKNN(k=NUM_NEIGHBORS)

    model.eval()
    device = next(model.parameters()).device
    num_layers = model.model.num_layers  # type: ignore

    all_predictions = []
    for atoms in atoms_list:
        graph = knn.convert(atoms)

        loader = NeighborLoader(
            graph,
            num_neighbors=[-1] * num_layers,
            batch_size=min(2**16, graph.num_nodes),  # type: ignore
            shuffle=False,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2,
        )

        graph_preds = []
        for batch in tqdm(loader, unit="batch", total=len(loader)):
            batch = batch.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model.predict_step(batch)
            graph_preds.append(out.to("cpu", non_blocking=True))

        torch.cuda.synchronize()
        all_predictions.append(torch.cat(graph_preds, dim=0))

    return all_predictions


def dump_outputs(
    predictions: list[Tensor],
    data_list: Iterable[DataCollection],
    pred_paths: Iterable[PathLike],
) -> None:
    for graph_preds, data, path in zip(predictions, data_list, pred_paths):
        path = Path(path)

        pred_array = graph_preds.detach().cpu().numpy()

        data.particles_.create_property("Prediction", data=pred_array)

        out_path = path.with_name(f"{path.stem}_predicted.extxyz")
        export_file(
            data,
            str(out_path),
            "xyz",
            columns=[
                "Particle Identifier",
                "Particle Type",
                "Position.X",
                "Position.Y",
                "Position.Z",
                "Prediction",
            ],
        )
        print(f"Saved predictions to {out_path}")


def main() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    seed_everything(DEFAULT_SEED)

    callbacks: list[Callback] = [
        ModelCheckpoint(monitor="val/loss", mode="min", every_n_epochs=1),
        RichModelSummary(max_depth=2),
    ]

    trainer = Trainer(
        max_epochs=EPOCHS,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    datamodule = LightningDataset(
        dataset_name="custom",
        lengths=(0.7, 0.2, 0.1),
        transforms=RandomPerturbation(std_range=(0.0, 0.05)),
        num_workers=8,
        batch_size=512,
        k=NUM_NEIGHBORS,
        use_imbalance_sampler=True,
    )

    num_classes = datamodule.num_classes
    metrics = {
        "f1": torchmetrics.F1Score(task="multiclass", num_classes=num_classes),
        "auroc": torchmetrics.AUROC(task="multiclass", num_classes=num_classes),
        "acc": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
    }

    if CKPT_NAME is None or not Path(CKPT_NAME).exists():
        with trainer.init_module():
            model = Module(
                model_name="PaiNN",
                num_classes=num_classes,
                compile=COMPILE,
                metrics=torchmetrics.MetricCollection(metrics),  # type: ignore
                warmup=400,
                lr=0.004678965862088063,
                max_iters=EPOCHS * len(datamodule.train_dataloader()),
                model_kwargs={
                    "num_radial": 8,
                    "hidden_channels": 32,
                    "num_layers": 2,
                    "dropout": 0.5,
                    "scale_factor": 1.0 / math.sqrt(NUM_NEIGHBORS),
                },
            )

        train(trainer, model, datamodule)
        # train_epoch(model, datamodule.train_dataloader())
    else:
        print(f"Loading checkpoint weights from: {CKPT_NAME}")
        with trainer.init_module(empty_init=True):
            model = Module.load_from_checkpoint(CKPT_NAME)

    if not TO_PREDICT:
        print("No files specified for prediction. Exiting.")
        return

    loaded_data: list[DataCollection] = [import_file(str(path)).compute() for path in TO_PREDICT]

    for data in loaded_data:
        positions = data.particles_.positions[...]
        data.particles_.positions_[...] = positions + np.random.normal(
            scale=0.02, size=positions.shape
        )

    predictions = inference(model, loaded_data)
    dump_outputs(predictions, loaded_data, TO_PREDICT)


if __name__ == "__main__":
    main()
