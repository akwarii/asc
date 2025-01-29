import gc
import pickle
from pathlib import Path
from typing import Any

import lightning as L
import optuna
import torch
import torchmetrics
from lightning.pytorch.callbacks import BatchSizeFinder, LearningRateFinder, RichProgressBar
from optuna import distributions
from optuna.trial import TrialState
from optuna_integration import PyTorchLightningPruningCallback
from src.constants import DEFAULT_SEED
from src.datamodule import LightningDataset
from src.module import Module
from src.transforms import LineGraph, RandomPerturbation

EPOCHS = 50
K_NEIGH = 12
BUDGET = 100
STUDY_NAME = f"cegann-k{K_NEIGH}"
STORAGE = f"sqlite:///{STUDY_NAME}.db"


def get_sampler_and_pruner(
    sampler_pkl_path: Path,
) -> tuple[optuna.samplers.BaseSampler, optuna.pruners.BasePruner]:
    if sampler_pkl_path.exists():
        with open(sampler_pkl_path, "rb") as f:
            sampler = pickle.load(f)
    else:
        sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.HyperbandPruner()

    return sampler, pruner


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


def trial_step(hparams: dict[str, Any]) -> None:
    trial = study.ask(hparams)
    report_trial_params(trial)

    # Check whether we already evaluated the sampled hyperparameters
    # If it exists, then use the existing value as trial duplicated the parameters.
    for t in reversed(trial.study.get_trials(deepcopy=False)):
        if trial.params == t.params and trial.number != t.number:
            print(f"Trial {trial.number} is a duplicate of trial {t.number}.")
            study.tell(trial, state=TrialState.PRUNED)
            return

    # Configure the Lightning Trainer
    trainer = L.Trainer(
        precision="16-mixed",
        enable_progress_bar=True,
        logger=True,
        enable_checkpointing=False,
        max_epochs=EPOCHS,
        log_every_n_steps=10,
        callbacks=[
            RichProgressBar(),
            BatchSizeFinder(init_val=32, steps_per_trial=10, max_trials=8),
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

    # Run the training loop and check if the trial was pruned
    state = optuna.trial.TrialState.COMPLETE
    try:
        trainer.fit(model, datamodule=datamodule)
    except optuna.TrialPruned:
        state = optuna.trial.TrialState.PRUNED
        print(f"Trial {trial.number} was pruned.")
    except (Exception, KeyboardInterrupt) as e:
        state = optuna.trial.TrialState.FAIL
        print(f"Trial {trial.number} failed: {e}")

    if state in [optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL]:
        study.tell(trial, state=state)
    else:
        val_acc = trainer.callback_metrics["val/acc"].item()
        study.tell(trial, val_acc, state)
        print(
            f"Trial {trial.number} finished with value: {val_acc} and parameters: {trial.params}"
        )

    # Make sure the memory is correctly released
    torch.cuda.empty_cache()
    torch._dynamo.reset()
    gc.collect()
    del model
    del trainer


if __name__ == "__main__":
    L.seed_everything(DEFAULT_SEED)

    sampler_pkl_path = Path(f"{STUDY_NAME}_sampler.pkl")
    sampler, pruner = get_sampler_and_pruner(sampler_pkl_path)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=STUDY_NAME,
        storage=STORAGE,
        load_if_exists=True,
    )

    hparam_distributions = {
        "edge_expansion_units": distributions.CategoricalDistribution([32, 64, 128, 256, 512]),
        "angle_expansion_units": distributions.CategoricalDistribution([32, 64, 128, 256, 512]),
        "num_radial": distributions.IntDistribution(20, 100),
        "n_bond_conv": distributions.IntDistribution(1, 6),
        "dropout": distributions.FloatDistribution(0.0, 0.8, step=0.1),
        "classification_units": distributions.CategoricalDistribution([32, 64, 128, 256, 512]),
        "classification_layers": distributions.IntDistribution(1, 4),
    }

    # Configure the Lightning Module. Only reload the dataset if there are no trials.
    # It means that the number of neighbors may have changed.
    if len(study.get_trials(deepcopy=False)) == 0:
        force_reload = True
    else:
        force_reload = False

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
        persistent_workers=False,  # optuna seems to spawn new persistent workers at each trial
    )

    # Hyperparameter optimization loop
    for _ in range(BUDGET):
        # Make sure the budget is shared across all processes
        if len(study.get_trials(deepcopy=False)) >= BUDGET:
            break

        trial_step(hparam_distributions)

        with open(sampler_pkl_path, "wb") as f:
            pickle.dump(study.sampler, f)

    # Report the statistics of the study
    report_statistics(study)
