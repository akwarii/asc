import torch
import torch_geometric.transforms as T
import torchmetrics
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateFinder,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import CSVLogger
from src.constants import DEFAULT_SEED
from src.datamodule import CEGANNLightningDataset
from src.datasets import CSG
from src.models import CEGANN
from src.module import CEGANNModule
from src.transforms import LineGraph, RandomPerturbation
from src.utils.dataset import random_split
from torch_geometric.loader import ImbalancedSampler

seed_everything(DEFAULT_SEED)

# Data management
dataset = CSG(
    pre_transform=LineGraph(),
    transform=T.Compose(
        [
            T.NormalizeFeatures(["x", "edge_attr"]),
            RandomPerturbation(),
        ]
    ),
    k=12,
    rcut=6.0,
    force_reload=False,
)
train_dataset, val_dataset, test_dataset = random_split(
    dataset=dataset,
    lengths=(0.7, 0.2, 0.1),
)
datamodule = CEGANNLightningDataset(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    batch_size=128,
    num_workers=5,
    sampler=ImbalancedSampler(torch.tensor([data.y[0].item() for data in train_dataset])),
)

module = CEGANNModule(
    model_name="cegann",
    model_kwargs={"n_classes": dataset.num_classes},
    compile=True,
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
    max_epochs=100,
    precision="16-mixed",
    callbacks=[
        # BatchSizeFinder(steps_per_trial=100),
        LearningRateFinder(min_lr=1e-5, max_lr=0.1, num_training_steps=5_000),
        StochasticWeightAveraging(swa_lrs=0.01),
    ],
    logger=CSVLogger(save_dir="."),
    deterministic=True,
)

trainer.fit(model=module, datamodule=datamodule)
