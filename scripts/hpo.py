import argparse
import warnings

import optuna
import optunahub
import torch
import torchmetrics
from fvcore.nn import FlopCountAnalysis
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, RichModelSummary
from optuna.trial import TrialState
from src import LightningDataset, Module
from src.constants import DEFAULT_SEED
from src.transforms import BoxStrain, LineGraph, RandomPerturbation
from src.transforms.line_graph import LineGraphData
from torch import Tensor

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
torch.serialization.add_safe_globals(
    [LineGraphData, LineGraph]
)  # , RandomPerturbation, BoxScaling, BoxShearing])
torch.set_float32_matmul_precision("high")


module = optunahub.load_module(package="samplers/auto_sampler")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Models hyperparameter optimization.")
    parser.add_argument(
        "--model",
        type=str,
        choices=("cegann", "mlp", "gat", "cegannv2", "painn"),
        default="cegannv2",
        help="Model to optimize the hyperparameters for.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=250,
        help="Number of trials to run for the optimization.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size to use for the training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs to train the model for.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("csg", "mp", "aflow", "gnome", "custom"),
        default="csg",
        help="Name of the dataset to use for the training.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///logs/hpo.db",
        help="URL to the database to store the optimization results.",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable model compilation.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Name of the Optuna study. If not provided, it will be set to <model_name>.",
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
        if value is None:
            print(f"    {param:<22}: None ({importance:.2%})")
        else:
            print(f"    {param:<22}: {value:<3} ({importance:.2%})")


def sample_hyperparameters(trial: optuna.Trial) -> dict:
    if MODEL_NAME == "cegann":
        _ = trial.suggest_categorical("edge_expansion_units", [64, 128, 256, 512])
        _ = trial.suggest_categorical("angle_expansion_units", [32, 64, 128, 256, 512])
        _ = trial.suggest_int("num_radial", 30, 120, step=10)
        _ = trial.suggest_int("n_bond_conv", 1, 6)
        _ = trial.suggest_float("dropout", 0.1, 0.6, step=0.1)
        _ = trial.suggest_categorical("classification_units", [64, 128, 256, 512])
        _ = trial.suggest_int("classification_layers", 1, 4)

    elif MODEL_NAME == "cegannv2":
        _ = trial.suggest_categorical("emb_num_radial", [4, 8, 16])
        _ = trial.suggest_categorical("emb_num_angular", [16, 32, 64])
        _ = trial.suggest_categorical("emb_num_channels", [32, 64, 128])
        _ = trial.suggest_int("emb_num_layers", 1, 2)
        _ = trial.suggest_categorical("conv_hidden_channels", [64, 128, 256])
        _ = trial.suggest_categorical("conv_node_out_channels", [64, 128, 256])
        _ = trial.suggest_categorical("conv_edge_out_channels", [32, 64, 128])
        _ = trial.suggest_int("conv_num_layers", 2, 4)
        _ = trial.suggest_categorical("conv_heads", [2, 4, 8])
        _ = trial.suggest_categorical("conv_concat", [True, False])
        # _ = trial.suggest_categorical("conv_residual", [True, False])
        _ = trial.suggest_float("dropout", 0.2, 0.5, step=0.1)
        # _ = trial.suggest_categorical("act", ["LeakyReLU", "SiLU"])

        _ = trial.suggest_float("lr", 5.0e-5, 5e-3, log=True)
        _ = trial.suggest_int("warmup", 0, 1000, step=100)
        _ = trial.suggest_int("k", 10, 16)

        # trial.set_user_attr("k", 16)
        # trial.set_user_attr("emb_num_radial", 4)
        # trial.set_user_attr("emb_num_angular", 64)
        # trial.set_user_attr("emb_num_channels", 128)
        # trial.set_user_attr("conv_hidden_channels", 256)
        # trial.set_user_attr("conv_node_out_channels", 256)
        # trial.set_user_attr("conv_edge_out_channels", 32)
        # trial.set_user_attr("conv_heads", 8)
        trial.set_user_attr("conv_residual", True)
        # trial.set_user_attr("conv_num_layers", 4)
        # trial.set_user_attr("dropout", 0.3)
        trial.set_user_attr("act", "SiLU")

    elif MODEL_NAME == "painn":
        import math

        _ = trial.suggest_categorical("num_radial", [4, 8, 16])
        _ = trial.suggest_categorical("hidden_channels", [32, 64, 128])
        _ = trial.suggest_int("num_layers", 2, 4)
        _ = trial.suggest_float("dropout", 0.1, 0.6, step=0.1)
        _ = trial.suggest_float("lr", 5.0e-5, 5.0e-3, log=True)
        _ = trial.suggest_int("warmup", 100, 500, step=100)
        _ = trial.suggest_int("k", 10, 20)
        trial.set_user_attr("scale_factor", 1.0 / math.sqrt(trial.params["k"]))

    elif MODEL_NAME == "mlp":
        _ = trial.suggest_int("num_layers", 3, 8)
        _ = trial.suggest_categorical("hidden_channels", [64, 128, 256, 512, 1024])
        _ = trial.suggest_int("num_radial", 30, 120, step=10)
        _ = trial.suggest_categorical("act", ["ReLU", "LeakyReLU", "SiLU", "ELU"])
        _ = trial.suggest_float("dropout", 0.1, 0.6, step=0.1)

    elif MODEL_NAME == "gat":
        _ = trial.suggest_categorical("hidden_channels", [64, 128, 256])
        _ = trial.suggest_int("num_layers", 1, 5)
        _ = trial.suggest_int("num_radial", 30, 120, step=10)
        _ = trial.suggest_float("dropout", 0.1, 0.6, step=0.1)
        _ = trial.suggest_categorical("act", ["ReLU", "LeakyReLU", "SiLU", "ELU"])
        _ = trial.suggest_categorical("norm", ["batch_norm", "layer_norm", None])
        _ = trial.suggest_categorical("heads", [1, 2, 4, 8])
        _ = trial.suggest_categorical("share_weights", [True, False])
        _ = trial.suggest_categorical("residual", [True, False])
        _ = trial.suggest_categorical("classification_units", [64, 128, 256, 512])
        _ = trial.suggest_int("classification_layers", 1, 4)

    else:
        raise NotImplementedError(f"HPO is not implemented for {MODEL_NAME} model.")

    # _ = trial.suggest_int("k", 10, 16, step=2)

    hparams = trial.params.copy()

    for key in trial.user_attrs.keys():
        hparams[key] = trial.user_attrs[key]

    return hparams


def objective(trial: optuna.Trial) -> tuple[float, float]:
    # Make sure the memory is correctly released
    torch.cuda.empty_cache()
    torch._dynamo.reset()

    # Sample the hyperparameters
    hparams = sample_hyperparameters(trial)
    report_trial_params(trial)

    # Check whether we already evaluated the sampled hyperparameters
    # If it exists, then use the existing value as trial duplicated the parameters.
    for t in reversed(trial.study.get_trials(deepcopy=False)):
        if trial.params == t.params and trial.number != t.number:
            if t.value is not None:
                raise optuna.exceptions.TrialPruned("Duplicated hyperparameters.")

    # Configure the Lightning Trainer
    trainer = Trainer(
        logger=True,
        precision="bf16-mixed",
        enable_progress_bar=True,
        enable_model_summary=False,
        enable_checkpointing=False,
        max_epochs=EPOCHS,
        callbacks=[
            RichModelSummary(),
            # PyTorchLightningPruningCallback(trial, monitor="val/f1"),
            EarlyStopping(monitor="val/loss", mode="min", patience=20, check_finite=True),
        ],
        deterministic=True,
    )

    # Configure the Lightning DataModule
    datamodule = LightningDataset(
        dataset_name=DATASET,
        lengths=(0.8, 0.2),
        use_imbalance_sampler=False,
        pre_transforms=LineGraph() if MODEL_NAME != "painn" else None,
        transforms=[
            RandomPerturbation(std_range=(0.0, 0.05)),
            BoxStrain(std_range=(0.0, 0.05)),
        ]
        if DATASET != "csg"
        else None,
        num_workers=5,
        k=hparams["k"],
        batch_size=BATCH_SIZE,
        persistent_workers=False,
    )
    num_classes = datamodule.num_classes

    # Configure the Lightning Module
    with trainer.init_module():
        model = Module(
            model_name=MODEL_NAME,
            compile=COMPILE,
            num_classes=num_classes,
            metrics=torchmetrics.MetricCollection(
                {
                    "f1": torchmetrics.F1Score(
                        task="multiclass",
                        num_classes=num_classes,
                    ),
                }
            ),
            warmup=hparams.pop("warmup", 100),
            lr=hparams.pop("lr", 1e-3),
            max_iters=trainer.max_epochs * len(datamodule.train_dataloader()),  # type: ignore
            model_kwargs=hparams,
        )

    # Log the hyperparameters and start the training
    if trainer.logger is not None:
        trainer.logger.log_hyperparams(trial.params)

    trainer.fit(model, datamodule=datamodule)

    # Evaluate the model on the validation set and compute the FLOPs
    device = model.device
    batch = next(iter(datamodule.val_dataloader()))
    inputs = (batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device))

    metrics = [
        trainer.callback_metrics.get("val/f1", 0.0),
        trainer.callback_metrics.get("val/loss", 0.0),
        FlopCountAnalysis(model, inputs).total(),
    ]

    for metric in metrics:
        if isinstance(metric, Tensor):
            metric = metric.item()

    del trainer
    del datamodule
    del model

    return (*metrics,)


if __name__ == "__main__":
    seed_everything(DEFAULT_SEED)

    args = parse_args()

    MODEL_NAME = args.model
    BUDGET = args.budget
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    DATASET = args.dataset
    STORAGE = args.storage
    COMPILE = not args.no_compile
    STUDY_NAME = args.study_name or MODEL_NAME

    storage = optuna.storages.RDBStorage(
        url=STORAGE,
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=3),
        engine_kwargs={"connect_args": {"timeout": 60.0}},
    )

    study = optuna.create_study(
        directions=["maximize", "minimize", "minimize"],
        sampler=module.AutoSampler(),
        pruner=optuna.pruners.NopPruner(),
        study_name=STUDY_NAME,
        storage=storage,
        load_if_exists=True,
    )
    study.set_metric_names(["val/f1", "val/loss", "flops"])

    study.optimize(
        objective,
        n_trials=BUDGET,
        callbacks=[optuna.study.MaxTrialsCallback(BUDGET)],
        gc_after_trial=True,
    )

    report_statistics(study)
