import optuna
import pytorch_lightning as pl
import torch_geometric.transforms as T
import torchmetrics
from src.datamodule import LightningDataset
from src.module import CEGANNModule
from src.transforms import LineGraph, RandomPerturbation
from torch import optim


def objective(trial: optuna.trial.Trial) -> float:
    n_conv_edge = trial.suggest_int("n_conv_edge", 2, 5)
    edge_expansion_units = trial.suggest_categorical("edge_expansion_units", [32, 64, 128, 256])
    angle_expansion_units = trial.suggest_categorical("angle_expansion_units", [32, 64, 128, 256])
    num_radial = trial.suggest_int("num_radial", 20, 100)
    k_neigh = trial.suggest_int("k_neigh", 8, 24)

    # Data management
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
        k=k_neigh,
        rcut=6.0,
        force_reload=False,
    )

    model = CEGANNModule(
        model_name="cegann",
        model_kwargs={
            "n_classes": datamodule.num_classes,
            "n_conv_edge": n_conv_edge,
            "edge_expansion_units": edge_expansion_units,
            "angle_expansion_units": angle_expansion_units,
            "rbf": {"num_radial": num_radial},
            "sbf": {"num_radial": num_radial},
        },
        optimizer=optim.AdamW,
        scheduler=optim.lr_scheduler.StepLR,
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

    trainer = pl.Trainer(check_val_every_n_epoch=5, max_epochs=10)
    trainer.fit(model=model, datamodule=datamodule)
    acc = trainer.test(datamodule=datamodule, ckpt_path="best")

    return acc[0]["test/acc"]


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=48)
