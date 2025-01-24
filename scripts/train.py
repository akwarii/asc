import torch_geometric.transforms as T
import torchmetrics
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateFinder, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from src.constants import DEFAULT_SEED
from src.datamodule import LightningDataset
from src.module import Module
from src.transforms import LineGraph, RandomPerturbation

seed_everything(DEFAULT_SEED)

trainer = Trainer(
    max_epochs=10,
    precision="16-mixed",
    callbacks=[
        # BatchSizeFinder(steps_per_trial=10, init_val=128),
        LearningRateFinder(min_lr=1e-5, max_lr=0.1),
        ModelCheckpoint(),
    ],
    logger=CSVLogger(save_dir="."),
    deterministic=True,
)

datamodule = LightningDataset(
    dataset_name="csg",
    lengths=(0.7, 0.2, 0.1),
    batch_size=1024,
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

model = Module(
    model_name="cegann",
    num_classes=datamodule.num_classes,
    compile=True,
    metrics=torchmetrics.MetricCollection(
        {
            "acc": torchmetrics.Accuracy(
                task="multiclass",
                num_classes=datamodule.num_classes,
            ),
        }
    ),
    warmup=100,
    max_iters=trainer.max_epochs * len(datamodule.train_dataloader()),  # type: ignore
)

trainer.fit(model=model, datamodule=datamodule)
trainer.validate(model=model, datamodule=datamodule)
trainer.test(model=model, datamodule=datamodule)
