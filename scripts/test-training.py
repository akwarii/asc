import torch
import torch_geometric.transforms as T
import torchmetrics
from pytorch_lightning import Trainer
from torch.utils.data import random_split
from torch_geometric.data.lightning import LightningDataset

from src.augmentation import (
    RandomDisplacement,
    RandomNodeDrop,
)
from src.datasets import CSG, CustomDataset
from src.models.cegann import CEGANN
from src.module import CEGANNModule

SEED = 42

# Data management
dataset = CSG(
# dataset = CustomDataset(
    transform=T.Compose(
        [
            T.NormalizeFeatures(["edge_dist"]),
            RandomDisplacement(p=0.2),
            # RandomNodeDrop(p=0.2),
        ]
    ),
    k=12,
    rcut=6.0,
)
train_dataset, val_dataset, test_dataset = random_split(
    dataset=dataset,
    lengths=(0.8, 0.1, 0.1),
    generator=torch.Generator().manual_seed(SEED),
)
datamodule = LightningDataset(
    train_dataset=train_dataset,  # type: ignore
    val_dataset=val_dataset,  # type: ignore
    test_dataset=test_dataset,  # type: ignore
    batch_size=64,
    num_workers=5,
)

model = CEGANN(
    gbf_bond={
        "start": 0.0,
        "stop": 1.0,
        "num_radial": 80,
    },
    gbf_angle={
        "start": 0.0,
        "stop": 1.0,
        "num_radial": 80,
    },
    n_classes=dataset.num_classes,
    edge_expansion_units=256,
    angle_expansion_units=256,
    n_conv_edge=2,
)

module = CEGANNModule(
    model=model,
    optimizer=torch.optim.AdamW,
    scheduler=torch.optim.lr_scheduler.StepLR,
    scheduler_params={"gamma": 0.5, "step_size": 100},
    metrics=torchmetrics.MetricCollection(
        {
            "acc": torchmetrics.Accuracy(
                task="multiclass",
                num_classes=dataset.num_classes,
            ),
        }
    ),
)

trainer = Trainer(
    # fast_dev_run=True,
    max_epochs=1,
    limit_train_batches=1000,
    precision="16-mixed",
)

trainer.fit(model=module, datamodule=datamodule)
