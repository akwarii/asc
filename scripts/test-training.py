import os

import lightning as L
import torch
import torch._dynamo.config
import torch.multiprocessing
import torchmetrics
import torchmetrics.classification
from lightning.pytorch.callbacks import BatchSizeFinder
from torch import optim
from torch.nn import CrossEntropyLoss
from torch_geometric.transforms import NormalizeFeatures

from src.augmentation import (
    RandomDisplacement,
    RandomExpansion,
    RandomNodeDrop,
)
from src.datamodule import CEGANNDataModule
from src.models.cegann import CEGANN
from src.module import CEGANNModule

torch.set_float32_matmul_precision("medium")
torch.cuda.empty_cache()
torch.multiprocessing.set_sharing_strategy("file_system")

cpu_count = os.cpu_count()

# Data management
transforms = [NormalizeFeatures(["edge_dist", "angle_cos"])]
augmenters = [
    RandomDisplacement(p=1.0),
    # RandomExpansion(p=0.05),
    # RandomNodeDrop(p=0.2),
]

datamodule = CEGANNDataModule(
    datasets="mp",
    transforms=transforms,
    augmentations=augmenters,
    num_workers=cpu_count - 1 if cpu_count is not None else 0,
    train_val_test_split=(0.8, 0.1, 0.1),
    batch_size=256,
    graph_kwargs={
        "k": 12,
    },
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
    n_classes=230,
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

# Training
trainer = L.Trainer(
    limit_train_batches=1000,
    limit_val_batches=100,
    max_epochs=20,
    callbacks=[BatchSizeFinder(init_val=256)],
    # log_every_n_steps=1,
    # fast_dev_run=True,
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
