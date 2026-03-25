import math
from pathlib import Path

import torch
import torchmetrics
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from line_profiler import profile
from ovito.data import DataCollection
from ovito.io import export_file, import_file
from src import LightningDataset, Module
from src.constants import DEFAULT_SEED
from src.graph import PeriodicKNN
from src.models.painn import PaiNN
from src.transforms import RandomPerturbation
from src.transforms.box_strain import BoxStrain
from src.typing import PathLike
from torch import Tensor
from torch_geometric.loader import NeighborLoader
from tqdm.auto import tqdm

torch.serialization.add_safe_globals([PaiNN])


EPOCHS = 30
NUM_NEIGHBORS = 20
COMPILE = True
# CKPT_NAME = "./lightning_logs/version_2/checkpoints/epoch=29-step=1020.ckpt"
CKPT_NAME = None
TO_PREDICT: list[PathLike] = [
    Path.home() / "THESE" / "TEST" / "Si_mixture_polycrystal" / "final.cfg",
]


def train(trainer: Trainer, model: Module, datamodule: LightningDataset) -> None:
    trainer.fit(model=model, datamodule=datamodule)
    trainer.validate(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)

    del trainer, datamodule


@torch.inference_mode()
@profile
def inference(model: Module, data: DataCollection) -> Tensor:
    """Run inference on a single DataCollection object and return the predicted class indices for
    each particle.

    Args:
        model: The trained PyTorch Lightning Module for prediction.
        data: The input DataCollection object containing the particle data to predict on.
    """
    model.eval()
    device = next(model.parameters()).device
    num_layers: int = model.model.num_layers  # type: ignore

    knn = PeriodicKNN(k=NUM_NEIGHBORS)
    graph = knn.convert(data)

    loader = NeighborLoader(
        graph,
        num_neighbors=[NUM_NEIGHBORS] * num_layers,
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
        with torch.autocast(device_type=device.type):
            out = model.predict_step(batch)
        graph_preds.append(out.to("cpu", non_blocking=True))

    torch.cuda.synchronize()
    predictions = torch.cat(graph_preds, dim=0)

    return predictions


def dump_outputs(
    preds: Tensor,
    data: DataCollection,
    fpath: PathLike,
) -> None:
    """Write predictions to an .extxyz file with a new property column named 'Prediction'.
    The output file will be saved in the same directory as the input file, with the
    "_predicted" suffix appended to the filename.

    Args:
        preds: Tensor containing the predicted class indices for each particle
        data: The original DataCollection object
        fpath: The file path of the original file
    """
    path = Path(fpath)
    out_path = path.with_name(f"{path.stem}_predicted.extxyz")

    pred_array = preds.detach().cpu().numpy()
    data.particles_.create_property("Prediction", data=pred_array)

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
    ]

    trainer = Trainer(
        max_epochs=EPOCHS,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    augmentations = [
        RandomPerturbation(std_range=(0.0, 0.05)),
        BoxStrain(std_range=(0.0, 0.05), directions="all"),
    ]

    datamodule = LightningDataset(
        dataset_name="custom",
        lengths=(0.7, 0.2, 0.1),
        root="./examples/silicon/data",
        transforms=augmentations,
        k=NUM_NEIGHBORS,
        num_workers=8,
        batch_size=512,
        use_imbalance_sampler=True,
    )

    num_classes = datamodule.num_classes
    metrics = [
        torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
        torchmetrics.F1Score(task="multiclass", num_classes=num_classes),
        torchmetrics.AUROC(task="multiclass", num_classes=num_classes),
        torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes),
    ]

    model = PaiNN(
        out_channels=num_classes,
        num_radial=8,
        hidden_channels=32,
        num_layers=2,
        dropout=0.5,
        scale_factor=1.0 / math.sqrt(NUM_NEIGHBORS),
    )

    if CKPT_NAME is None or not Path(CKPT_NAME).exists():
        with trainer.init_module():
            module = Module(
                model=model,
                metrics=metrics,
                compile=COMPILE,
                warmup=400,
                lr=0.004678965862088063,
                max_iters=EPOCHS * len(datamodule.train_dataloader()),
            )

        train(trainer, module, datamodule)
    else:
        print(f"Loading checkpoint weights from: {CKPT_NAME}")
        with trainer.init_module(empty_init=True):
            module = Module.load_from_checkpoint(CKPT_NAME, weights_only=False, model=model)

    if TO_PREDICT:
        for path in tqdm(TO_PREDICT):
            data = import_file(path).compute()
            preds = inference(module, data)
            dump_outputs(preds, data, path)
    else:
        print("No files specified for prediction.")


if __name__ == "__main__":
    main()
