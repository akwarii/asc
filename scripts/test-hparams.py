import optuna
import pytorch_lightning as pl
import torch_geometric.transforms as T
import torchmetrics
from src.datamodule import CEGANNLightningDataset
from src.datasets import CSG
from src.models.cegann import CEGANN
from src.module import CEGANNModule
from src.transforms import LineGraph, RandomPerturbation
from torch import optim
from torch.utils.data import random_split


def objective(trial: optuna.trial.Trial) -> float:
    n_conv_edge = trial.suggest_int("n_conv_edge", 2, 5)
    edge_expansion_units = trial.suggest_categorical("edge_expansion_units", [32, 64, 128, 256])
    angle_expansion_units = trial.suggest_categorical("angle_expansion_units", [32, 64, 128, 256])
    num_radial = trial.suggest_int("num_radial", 20, 100)
    k_neigh = trial.suggest_int("k_neigh", 8, 24)

    # Data management
    dataset = CSG(
        pre_transform=LineGraph(),
        transform=T.Compose(
            [
                T.NormalizeFeatures(["x", "edge_attr"]),
                RandomPerturbation(),
            ]
        ),
        k=k_neigh,
        rcut=6.0,
        force_reload=True,
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
    )
    # Model
    model = CEGANN(
        rbf={"num_radial": num_radial},
        sbf={"num_radial": num_radial},
        n_classes=dataset.num_classes,
        edge_expansion_units=edge_expansion_units,
        angle_expansion_units=angle_expansion_units,
        n_bond_conv=n_conv_edge,
    )
    module = CEGANNModule(
        model=model,
        compile=True,
        optimizer=optim.AdamW,
        scheduler=optim.lr_scheduler.StepLR,
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

    trainer = pl.Trainer(check_val_every_n_epoch=5, max_epochs=10)
    trainer.fit(model=module, datamodule=datamodule)
    acc = trainer.test(datamodule=datamodule, ckpt_path="best")

    return acc[0]["test/acc"]


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=48)
