# Based on
# https://lightning.ai/docs/pytorch/stable/starter/introduction.html
import os
import lightning as L
import torch
import torch._dynamo.config
import torch.multiprocessing
import torchmetrics
from torch import optim
from torch.nn import CrossEntropyLoss
from torch_geometric.transforms import NormalizeFeatures
import torchmetrics.classification

from src.augmentation import (
    RandomDisplacement,
    RandomExpansion,
    RandomNodeDrop,
)
from src.datamodule import CEGANNDataModule
from src.module import CEGANNModule
from src.models.cegann import CEGANN


torch.set_float32_matmul_precision("medium")
torch.cuda.empty_cache()
torch.multiprocessing.set_sharing_strategy("file_system")

cpu_count = os.cpu_count()

# Data management
transforms = [NormalizeFeatures(["edge_dist", "angle_cos"])]
augmenters = [RandomDisplacement(p=0.1), RandomExpansion(p=0.05), RandomNodeDrop(p=0.2)]

datamodule = CEGANNDataModule(
    datasets="mp",
    # datasets=["mp", "aflow", "csg", "gnome"],
    # datasets="custom", # test this (needs the following line also)
    # origin_dir="/home/dbissuel/Documents/softs/sourcecodes/CEGANN/pretrained/spacegroup/",
    transforms=transforms,
    augmentations=augmenters,
    num_workers=cpu_count - 1 if cpu_count is not None else 0,
    train_val_test_split=(0.8, 0.1, 0.1),
    batch_size=32,
    graph_kwargs={"k": 2, },
)
n_classes = datamodule.num_classes or 230

datamodule.prepare_data()
datamodule.setup(stage="fit")

train_loader = datamodule.train_dataloader()
test_loader = datamodule.test_dataloader()
val_loader = datamodule.val_dataloader()

# Model
model = CEGANN(
    gbf_bond={
        "start": 0.0,
        "stop": 1.0,
        "num_radial": 40,  # DB
    },
    gbf_angle={
        "start": 0.0,
        "stop": 1.0,
        "num_radial": 40,  # DB
    },
    n_classes=n_classes,
    edge_expansion_units=64,  # OG paper 256
    angle_expansion_units=64,  # OG paper 256
    n_conv_edge=2,
)

optimizer = optim.Adam
# optimizer = optim.SGD

# scheduler = optim.lr_scheduler.LinearLR
scheduler = optim.lr_scheduler.StepLR
# scheduler = optim.lr_scheduler.ExponentialLR
scheduler_params = {"gamma": 0.5, "step_size": 100}

criterion = CrossEntropyLoss(reduction="mean")

metrics = torchmetrics.MetricCollection(
    {
        "accuracy": torchmetrics.Accuracy(
            task="multiclass",
            num_classes=n_classes,
        ),
    }
)

module = CEGANNModule(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    scheduler_params=scheduler_params,
    criterion=criterion,
    metrics=metrics,
    compile=False,
)

# Training
trainer = L.Trainer(
    # limit_train_batches=0.01,
    # max_epochs=1,
    fast_dev_run=True,
    # check_val_every_n_epoch=5,
    # max_epochs=200,
)
trainer.fit(
    model=module,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)
# out = trainer.test(dataloaders=test_loader, ckpt_path="best")
# print("After testing :", out)

# Prediction
# datamodule.setup(stage="predict")
# pred_loader = datamodule.predict_dataloader()
# acc = trainer.predict(model=module,
#                       dataloaders=pred_loader,
#                       return_predictions=True,
#                       ckpt_path='best',)
# datamodule.teardown(stage="predict")
