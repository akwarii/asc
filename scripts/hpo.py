import argparse

import lightning as L
import optuna
import torch
import torchmetrics
from lightning.pytorch.callbacks import LearningRateFinder
from optuna.trial import TrialState
from optuna_integration import PyTorchLightningPruningCallback
from src.constants import DEFAULT_SEED
from src.datamodule import LightningDataset
from src.module import Module
from src.transforms import LineGraph, RandomPerturbation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for the CEGANN model."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["cegann", "mlp", "gat"],
        default="cegann",
        help="Model to optimize the hyperparameters for.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=100,
        help="Number of trials to run for the optimization.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs to train the model for.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///hpo.db",
        help="URL to the database to store the optimization results.",
    )
    return parser.parse_args()


def report_trial_params(trial: optuna.trial.Trial) -> None:
    print(f"Trial {trial.number}/{BUDGET}:")
    print("  Parameters: ")
    for param, value in trial.params.items():
        print(f"  {param}: {value}")


def report_statistics(study: optuna.study.Study) -> None:
    prunned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    failed_trials = study.get_trials(deepcopy=False, states=[TrialState.FAIL])
    best_trial = study.best_trial

    print("Study statistics: ")
    print(f"  Number of trials: {len(study.trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")
    print(f"  Number of pruned trials: {len(prunned_trials)}")
    print(f"  Number of failed trials: {len(failed_trials)}")

    print("Best trial:")
    print(f"  Value: {best_trial.value}")
    print("  Params (importance): ")
    hparams_importance = optuna.importance.get_param_importances(study)
    for (param, value), (_, importance) in zip(
        best_trial.params.items(), hparams_importance.items()
    ):
        print(f"    {param:<22}: {value:<3} ({importance:.2%})")


def sample_hyperparameters(trial: optuna.Trial) -> dict:
    if MODEL_NAME == "cegann":
        _ = trial.suggest_categorical("edge_expansion_units", [64, 128, 256, 512])
        _ = trial.suggest_categorical("angle_expansion_units", [32, 64, 128, 256, 512])
        _ = trial.suggest_int("num_radial", 60, 120)
        _ = trial.suggest_int("n_bond_conv", 1, 6)
        _ = trial.suggest_float("dropout", 0.2, 0.8, step=0.1)
        _ = trial.suggest_categorical("classification_units", [32, 64, 128, 256, 512])
        _ = trial.suggest_int("classification_layers", 1, 4)
        _ = trial.suggest_int("k", 10, 16, step=2)

    elif MODEL_NAME == "mlp":
        _ = trial.suggest_int("num_layers", 3, 8)
        _ = trial.suggest_categorical("hidden_channels", [64, 128, 256, 512, 1024])
        _ = trial.suggest_int("num_radial", 25, 120)
        _ = trial.suggest_categorical("act", ["ReLU", "LeakyReLU", "SiLU", "ELU"])
        _ = trial.suggest_float("dropout", 0.2, 0.8, step=0.1)
        _ = trial.suggest_int("k", 10, 16, step=2)

    elif MODEL_NAME == "gat":
        _ = trial.suggest_categorical("hidden_channels", [64, 128, 256, 512])
        _ = trial.suggest_int("num_layers", 1, 6)
        _ = trial.suggest_int("num_radial", 25, 120)
        _ = trial.suggest_float("dropout", 0.2, 0.8, step=0.1)
        _ = trial.suggest_categorical("act", ["ReLU", "LeakyReLU", "SiLU", "ELU"])
        _ = trial.suggest_categorical("norm", ["batch_norm", "layer_norm", None])
        _ = trial.suggest_categorical("heads", [1, 2, 4, 8])
        _ = trial.suggest_float("negative_slope", 0.1, 0.3, step=0.1)
        _ = trial.suggest_categorical("share_weights", [True, False])
        _ = trial.suggest_categorical("residual", [True, False])
        _ = trial.suggest_categorical("classification_units", [32, 64, 128, 256, 512])
        _ = trial.suggest_int("classification_layers", 1, 4)

    else:
        raise NotImplementedError(f"HPO is not implemented for {MODEL_NAME} models.")

    hparams = trial.params.copy()

    if MODEL_NAME in ("cegann", "mlp"):
        hparams["in_channels"] = -1

    return hparams


def objective(trial: optuna.Trial) -> float:
    # Make sure the memory is correctly released
    torch.cuda.empty_cache()
    torch._dynamo.reset()

    # Sample the hyperparameters
    hparams = sample_hyperparameters(trial)
    k_neigh = hparams.pop("k")

    report_trial_params(trial)

    # Check whether we already evaluated the sampled hyperparameters
    # If it exists, then use the existing value as trial duplicated the parameters.
    for t in reversed(trial.study.get_trials(deepcopy=False)):
        if trial.params == t.params and trial.number != t.number:
            if t.value is not None:
                return t.value

    # Configure the Lightning DataModule
    datamodule = LightningDataset(
        dataset_name="csg",
        lengths=(0.8, 0.2),
        use_imbalance_sampler=True,
        pre_transforms=LineGraph(),
        transforms=[
            RandomPerturbation(std=0.1),
        ],
        num_workers=5,
        k=k_neigh,
        batch_size=256,
        persistent_workers=False,
    )
    num_classes = datamodule.num_classes

    # Configure the Lightning Trainer
    trainer = L.Trainer(
        precision="16-mixed",
        enable_progress_bar=False,
        logger=True,
        enable_checkpointing=False,
        max_epochs=EPOCHS,
        log_every_n_steps=50,
        callbacks=[
            LearningRateFinder(min_lr=1e-5, max_lr=0.1),
            PyTorchLightningPruningCallback(trial, monitor="val/f1"),
        ],
    )

    # Configure the Lightning Module
    with trainer.init_module():
        model = Module(
            model_name=MODEL_NAME,
            num_classes=num_classes,
            metrics=torchmetrics.MetricCollection(
                {
                    "f1": torchmetrics.F1Score(
                        task="multiclass",
                        num_classes=num_classes,
                    ),
                }
            ),
            warmup=100,
            max_iters=trainer.max_epochs * len(datamodule.train_dataloader()),  # type: ignore
            model_kwargs=hparams,
        )

    # Log the hyperparameters and start the training
    if trainer.logger is not None:
        trainer.logger.log_hyperparams(trial.params)

    trainer.fit(model, datamodule=datamodule)

    metric = trainer.callback_metrics["val/f1"].item()

    del trainer
    del datamodule
    del model

    return metric


if __name__ == "__main__":
    L.seed_everything(DEFAULT_SEED)

    args = parse_args()

    MODEL_NAME = args.model
    BUDGET = args.budget
    EPOCHS = args.epochs
    STORAGE = args.storage

    storage = optuna.storages.RDBStorage(
        url=STORAGE,
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=3),
        engine_kwargs={"connect_args": {"timeout": 20.0}},
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(),
        study_name=MODEL_NAME,
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=BUDGET,
        callbacks=[optuna.study.MaxTrialsCallback(BUDGET, states=None)],
        gc_after_trial=True,
    )

    report_statistics(study)
