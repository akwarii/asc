import torch
import torch_geometric.transforms as T
import torchmetrics
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from src.constants import DEFAULT_SEED
from src.datamodule import LightningDataset
from src.module import CEGANNModule
from src.transforms import LineGraph, RandomPerturbation

seed_everything(DEFAULT_SEED)

datamodule = LightningDataset(
    dataset_name="custom",
    lengths=(0.7, 0.2, 0.1),
    batch_size=8,
    use_imbalance_sampler=True,
    pre_transforms=LineGraph(),
    transforms=[
        T.NormalizeFeatures(["x", "edge_attr"]),
        RandomPerturbation(),
    ],
    num_workers=5,
    k=12,
    rcut=6.0,
    force_reload=False,
)

model = CEGANNModule(
    model_name="cegann",
    model_kwargs={"n_classes": datamodule.num_classes},
    optimizer=torch.optim.AdamW,
    scheduler=torch.optim.lr_scheduler.StepLR,
    scheduler_params={"gamma": 0.5, "step_size": 100},
    metrics=torchmetrics.MetricCollection(
        {
            "acc": torchmetrics.Accuracy(
                task="multiclass",
                num_classes=datamodule.num_classes,
            ),
        }
    ),
)
# model = torch.compile(model)

trainer = Trainer(
    # fast_dev_run=True,
    max_epochs=1,
    precision="16-mixed",
    callbacks=[
        # pl_callbacks.BatchSizeFinder(steps_per_trial=100),
        # pl_callbacks.LearningRateFinder(min_lr=1e-5, max_lr=0.1, num_training_steps=5_000),
        # pl_callbacks.StochasticWeightAveraging(swa_lrs=0.01),
        # pl_callbacks.ModelCheckpoint(),
    ],
    logger=CSVLogger(save_dir="."),
    deterministic=True,
)

trainer.fit(model=model, datamodule=datamodule)  # type: ignore
trainer.validate(model=model, datamodule=datamodule)  # type: ignore
trainer.test(model=model, datamodule=datamodule)  # type: ignore
