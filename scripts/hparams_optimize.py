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

EPOCHS = 30
K_NEIGH = 12
BUDGET = 100
STUDY_NAME = f"cegann-k{K_NEIGH}"
STORAGE_URL = f"sqlite:///{STUDY_NAME}.db"


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


def objective(trial: optuna.Trial) -> float:
    # Make sure the memory is correctly released
    torch.cuda.empty_cache()
    torch._dynamo.reset()

    # Sample the hyperparameters
    _ = trial.suggest_categorical("edge_expansion_units", [32, 64, 128, 256, 512])
    _ = trial.suggest_categorical("angle_expansion_units", [32, 64, 128, 256, 512])
    _ = trial.suggest_int("num_radial", 20, 100)
    _ = trial.suggest_int("n_bond_conv", 1, 6)
    _ = trial.suggest_float("dropout", 0.0, 0.8, step=0.1)
    _ = trial.suggest_categorical("classification_units", [32, 64, 128, 256, 512])
    _ = trial.suggest_int("classification_layers", 1, 4)

    report_trial_params(trial)

    # Check whether we already evaluated the sampled hyperparameters
    # If it exists, then use the existing value as trial duplicated the parameters.
    for t in reversed(trial.study.get_trials(deepcopy=False)):
        if trial.params == t.params and trial.number != t.number:
            if t.value is not None:
                return t.value

    # Only reload the dataset if there are no trials.
    # It means that the number of neighbors may have changed.
    force_reload = False
    if trial.number == 0:
        force_reload = True

    datamodule = LightningDataset(
        dataset_name="csg",
        lengths=(0.8, 0.2),
        use_imbalance_sampler=True,
        pre_transforms=LineGraph(),
        transforms=[
            RandomPerturbation(std=0.1),
        ],
        num_workers=5,
        k=K_NEIGH,
        rcut=6.0,
        force_reload=force_reload,
        batch_size=512,
    )

    # Configure the Lightning Trainer
    trainer = L.Trainer(
        precision="16-mixed",
        enable_progress_bar=True,
        logger=True,
        enable_checkpointing=False,
        max_epochs=EPOCHS,
        log_every_n_steps=10,
        callbacks=[
            LearningRateFinder(min_lr=1e-5, max_lr=0.1),
            PyTorchLightningPruningCallback(trial, monitor="val/acc"),
        ],
    )

    # Configure the Lightning Module
    with trainer.init_module():
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
            model_kwargs=trial.params,
        )

    # Log the hyperparameters and start the training
    if trainer.logger is not None:
        trainer.logger.log_hyperparams(trial.params)

    trainer.fit(model, datamodule=datamodule)
    val_acc = trainer.callback_metrics["val/acc"].item()

    return val_acc


if __name__ == "__main__":
    L.seed_everything(DEFAULT_SEED)

    storage = optuna.storages.RDBStorage(
        url=STORAGE_URL,
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=3),
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(),
        study_name=STUDY_NAME,
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=BUDGET,
        callbacks=[optuna.study.MaxTrialsCallback(BUDGET)],
    )

    report_statistics(study)
