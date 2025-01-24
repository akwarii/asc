import pickle
from pathlib import Path

import lightning as L
import optuna
import torch_geometric.transforms as T
import torchmetrics
from lightning.pytorch.callbacks import BatchSizeFinder, LearningRateFinder
from optuna.importance import get_param_importances
from optuna.trial import TrialState
from optuna_integration import PyTorchLightningPruningCallback
from src.constants import DEFAULT_SEED
from src.datamodule import LightningDataset
from src.module import Module
from src.transforms import LineGraph, RandomPerturbation

EPOCHS = 50


def objective(trial: optuna.trial.Trial) -> float:
    k_neigh = trial.suggest_int("k_neigh", 6, 20)
    edge_units = trial.suggest_categorical("edge_units", [32, 64, 128, 256, 512])
    angle_units = trial.suggest_categorical("angle_units", [32, 64, 128, 256, 512])
    num_radial = trial.suggest_int("num_radial", 20, 100)
    num_conv_edge = trial.suggest_int("num_conv_edge", 1, 6)
    dropout = trial.suggest_float("dropout", 0.1, 0.8, step=0.1)
    classifier_units = trial.suggest_categorical("classifier_units", [32, 64, 128, 256, 512])
    classifier_layers = trial.suggest_int("classifier_layers", 1, 4)

    hyperparameters = {
        "k_neigh": k_neigh,
        "edge_expansion_units": edge_units,
        "angle_expansion_units": angle_units,
        "num_radial": num_radial,
        "n_bond_conv": num_conv_edge,
        "dropout": dropout,
        "classification_units": classifier_units,
        "classification_layers": classifier_layers,
    }

    # Check whether we already evaluated the sampled hyperparameters
    # If it exists, then use the existing value as trial duplicated the parameters.
    states_to_consider = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
    trials_to_consider = trial.study.get_trials(deepcopy=False, states=states_to_consider)
    for t in reversed(trials_to_consider):
        if trial.params == t.params:
            return t.value if t.value is not None else 0.0

    # Configure the Lightning Trainer
    trainer = L.Trainer(
        logger=True,
        enable_checkpointing=False,
        max_epochs=EPOCHS,
        callbacks=[
            BatchSizeFinder(steps_per_trial=10),
            LearningRateFinder(min_lr=1e-5, max_lr=0.1),
            PyTorchLightningPruningCallback(trial, monitor="val/acc"),
        ],
    )

    # Configure the Lightning Module
    datamodule = LightningDataset(
        dataset_name="csg",
        lengths=(0.8, 0.2),
        use_imbalance_sampler=True,
        pre_transforms=LineGraph(),
        transforms=[
            T.NormalizeFeatures(["x", "edge_attr"]),
            RandomPerturbation(std=0.05),
        ],
        num_workers=5,
        k=k_neigh,
        rcut=6.0,
        force_reload=True,
    )

    # Configure the Lightning Module
    model_kwargs = hyperparameters.copy()
    model_kwargs.pop("k_neigh", None)

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
        model_kwargs=model_kwargs,
    )

    # Log the hyperparameters and start the training
    if trainer.logger is not None:
        trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["val/acc"].item()


if __name__ == "__main__":
    L.seed_everything(DEFAULT_SEED)

    study_name = "cegann-00"
    storage = f"sqlite:///{study_name}.db"

    sampler_pkl_path = Path(f"{study_name}_sampler.pkl")
    if sampler_pkl_path.exists():
        with open(sampler_pkl_path, "rb") as f:
            sampler = pickle.load(f)
    else:
        sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.HyperbandPruner()

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="cegann-00",
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)

    with open(sampler_pkl_path, "wb") as f:
        pickle.dump(study.sampler, f)

    print(f"Number of finished trials: {len(study.trials)}")

    best_trial = study.best_trial

    print("Best trial:")
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    hparams_importance = get_param_importances(study)
    for (param, value), (_, importance) in zip(
        best_trial.params.items(), hparams_importance.items()
    ):
        print(f"    {param}: {value} ({importance:.2f})")

    study.trials_dataframe().to_csv("study.csv")
