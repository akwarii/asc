import math

import torch
import torchmetrics
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, RichModelSummary
from src import LightningDataset, Module
from src.constants import DEFAULT_SEED
from src.transforms import RandomPerturbation

EPOCHS = 100
NUM_NEIGHBORS = 12


def main() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    seed_everything(DEFAULT_SEED)

    callbacks: list[Callback] = [
        ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=3, every_n_epochs=1),
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
        dataset_name="csg",
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
            compile=True,
            metrics=torchmetrics.MetricCollection(metrics),  # type: ignore
            warmup=700,
            lr=6e-3,
            max_iters=EPOCHS * len(datamodule.train_dataloader()),
            model_kwargs={
                "num_radial": 4,
                "hidden_channels": 128,
                "num_layers": 4,
                "dropout": 0.3,
                "scale_factor": 1. / math.sqrt(NUM_NEIGHBORS),
            },
        )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.validate(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
