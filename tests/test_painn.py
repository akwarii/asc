import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torchmetrics
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, RichModelSummary
from ovito.data import DataCollection
from ovito.io import export_file, import_file
from src import LightningDataset, Module
from src.constants import DEFAULT_SEED
from src.graph import PeriodicKNN
from src.transforms import RandomPerturbation
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from tqdm.auto import tqdm

torch.serialization.add_safe_globals([RandomPerturbation])

EPOCHS = 50
NUM_NEIGHBORS = 20
COMPILE = False
CKPT_NAME = (
    Path(".") / "lightning_logs" / "version_45927470" / "checkpoints" / "epoch=45-step=1564.ckpt"
)
TO_PREDICT = [
    Path.home() / "THESE" / "TEST" / "Si_diamond_polycrystal" / "polycrystal.xyz",
]


def train(trainer: Trainer, model: Module, datamodule: LightningDataset) -> None:
    trainer.fit(model=model, datamodule=datamodule)
    trainer.validate(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)

    del trainer, datamodule


def _convert_to_graphs(atoms_list: Iterable[DataCollection]) -> list[Data]:
    knn = PeriodicKNN(k=NUM_NEIGHBORS)

    graph_list = []
    for atoms in atoms_list:
        data = knn.convert(atoms)
        graph_list.append(data)

    return graph_list


@torch.inference_mode()
def inference(model: Module, atoms_list: Iterable[DataCollection]) -> list[torch.Tensor]:
    to_predict = _convert_to_graphs(atoms_list)
    num_layers = model.model.num_layers  # type: ignore

    model.eval()
    device = next(model.parameters()).device

    all_predictions = []
    for graph in to_predict:
        loader = NeighborLoader(
            graph,
            num_neighbors=[-1] * num_layers,
            batch_size=2**16,
            shuffle=False,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2,
        )

        graph_preds = []
        for batch in tqdm(loader, unit="batch", total=len(loader)):
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model.predict_step(batch)
            graph_preds.append(out.to("cpu", non_blocking=True))

        torch.cuda.synchronize()
        all_predictions.append(torch.cat(graph_preds, dim=0))

    return all_predictions


def dump_outputs(
    predictions: list[torch.Tensor],
    data_list: Iterable[DataCollection],
    pred_paths: Iterable[Path],
) -> None:
    for graph_preds, data, path in zip(predictions, data_list, pred_paths):
        pred_array = graph_preds.detach().cpu().numpy()

        # 1. Attach custom property natively in OVITO
        data.particles_.create_property("Prediction", data=pred_array)

        # 2. Fast C++ export
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


def clean_state_dict(
    state_dict: dict[str, torch.Tensor], compile: bool
) -> dict[str, torch.Tensor]:
    """Removes the '_orig_mod.' prefix added by torch.compile from state dict keys."""
    if compile:
        return state_dict

    clean_dict = {}
    for key, val in state_dict.items():
        new_key = key.replace("_orig_mod.", "")
        clean_dict[new_key] = val
    return clean_dict


def main() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    seed_everything(DEFAULT_SEED)

    callbacks: list[Callback] = [
        ModelCheckpoint(monitor="val/loss", mode="min", every_n_epochs=1),
        RichModelSummary(max_depth=-1),
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
        transforms=RandomPerturbation(std=0.1),
        num_workers=8,
        batch_size=512,
        k=NUM_NEIGHBORS,
    )

    num_classes = datamodule.num_classes
    metrics = {
        "f1": torchmetrics.F1Score(task="multiclass", num_classes=num_classes),
        "auroc": torchmetrics.AUROC(task="multiclass", num_classes=num_classes),
        "acc": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
    }

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

    if CKPT_NAME is None or not Path(CKPT_NAME).exists():
        train(trainer, model, datamodule)
    else:
        print(f"Loading checkpoint weights from: {CKPT_NAME}")
        checkpoint = torch.load(CKPT_NAME, map_location="cpu", weights_only=False)
        model.load_state_dict(clean_state_dict(checkpoint["state_dict"], compile=COMPILE))

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
