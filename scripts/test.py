import torch
import torchmetrics
from lightning import Trainer, seed_everything
from src import LightningDataset, Module
from src.constants import DEFAULT_SEED
from src.datasets import CustomDataset
from src.transforms import LineGraph

CKPT = "lightning_logs/version_45927073/checkpoints/epoch=101-step=131378.ckpt"
BATCH_SIZE = 1


def main() -> None:
    seed_everything(DEFAULT_SEED)

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    ckpt_params = ckpt["hyper_parameters"] | ckpt["datamodule_hyper_parameters"]

    datamodule = LightningDataset(
        pred_dataset=CustomDataset(root="data/custom", k=ckpt_params["k"]),
        num_workers=4,
        batch_size=BATCH_SIZE,
        pre_transforms=LineGraph(),
        force_reload=True,
    )

    metrics = torchmetrics.MetricCollection(
        {
            "f1": torchmetrics.F1Score(task="multiclass", num_classes=ckpt_params["num_classes"]),
            "auroc": torchmetrics.AUROC(task="multiclass", num_classes=ckpt_params["num_classes"]),
            "acc": torchmetrics.Accuracy(
                task="multiclass", num_classes=ckpt_params["num_classes"]
            ),
            "confmat": torchmetrics.ConfusionMatrix(
                task="multiclass", num_classes=ckpt_params["num_classes"]
            ),
        }
    )

    trainer = Trainer(
        precision="16-mixed" if torch.cuda.is_available() else 32,
        deterministic=False,
        enable_progress_bar=True,
    )

    with trainer.init_module():
        model = Module.load_from_checkpoint(checkpoint_path=CKPT, metrics=metrics)

    _ = trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
