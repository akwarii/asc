import os

import torch
import torch._dynamo.config
import torch.multiprocessing
import torchmetrics
import torchmetrics.classification
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import SingleDeviceStrategy
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import random_split
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.transforms import Compose, NormalizeFeatures

from src.augmentation import (
    RandomDisplacement,
    RandomExpansion,
    RandomNodeDrop,
)
from src.datasets import MaterialProject
from src.models.cegann import CEGANN
from src.module import CEGANNModule

torch.set_float32_matmul_precision("medium")
torch.cuda.empty_cache()
torch.multiprocessing.set_sharing_strategy("file_system")

CPU_COUNT = os.cpu_count()
SEED = 42

# Data management
transform = Compose(
    [
        NormalizeFeatures(["edge_dist", "angle_cos"]),
        RandomDisplacement(p=1.0),
        RandomExpansion(p=0.05),
        RandomNodeDrop(p=0.2),
    ]
)

dataset = MaterialProject(
    # transform=transform,
    k=12,
    rcut=6.0,
)
# train_dataset, val_dataset, test_dataset = random_split(
#     dataset=dataset,
#     lengths=(0.8, 0.1, 0.1),
#     generator=torch.Generator().manual_seed(SEED),
# )
train_dataset = dataset[: int(0.8 * len(dataset))]
val_dataset = dataset[int(0.8 * len(dataset)) : int(0.9 * len(dataset))]
test_dataset = dataset[int(0.9 * len(dataset)) :]

datamodule = LightningDataset(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    batch_size=256,
    num_workers=CPU_COUNT - 1,
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
    edge_expansion_units=256,  # OG paper 256
    angle_expansion_units=256,  # OG paper 256
    n_conv_edge=2,
)

module = CEGANNModule(
    model=model,
    optimizer=optim.AdamW,
    scheduler=optim.lr_scheduler.StepLR,
    scheduler_params={"gamma": 0.5, "step_size": 100},
    criterion=CrossEntropyLoss(reduction="mean"),
    metrics=torchmetrics.MetricCollection(
        {
            "accuracy": torchmetrics.Accuracy(
                task="multiclass",
                num_classes=230,
            ),
        }
    ),
    compile=False,
)

strategy = SingleDeviceStrategy("cuda:0")
trainer = Trainer(
    fast_dev_run=True,
    strategy=strategy,
    devices=1,
)

trainer.fit(model=module, datamodule=datamodule)
trainer.test(model=module, datamodule=datamodule, ckpt_path="best")
acc = trainer.predict(
    model=module,
    datamodule=datamodule,
    return_predictions=True,
    ckpt_path="best",
)
print("Prediction :", acc)
