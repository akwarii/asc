# Hyperparameters optimization
import optuna

# Data management
from src.data.cegann_datamodule import CEGANNDataModule
from torch_geometric.transforms import NormalizeFeatures

# Model
from torch import optim
from torch.nn import CrossEntropyLoss
import torchmetrics
from src.models.components.cegann import CEGANN
from src.models.cegann_module import CEGANNModule

# Training
import lightning as L

# ON MY MACHINE ONLY
import torch
torch.set_float32_matmul_precision('medium')

# Necessarry for AFLOW and custom only
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def objective(trial) :

    # Counted by hand, number of target values
    # n_classes = 230
    n_classes = 8
    
    # Hyperparameters to optimize
    n_conv_edge = trial.suggest_int("n_conv_edge", 2, 5)
    edge_expansion_units = trial.suggest_categorical("edge_expansion_units", [32, 64, 128, 256])
    angle_expansion_units = trial.suggest_categorical("angle_expansion_units", [32, 64, 128, 256])
    num_radial = trial.suggest_int("num_radial", 20, 100)
    k_neigh = trial.suggest_int("k_neigh", 8, 24)

    # Defining the model
    
    # Data management
    transforms=[NormalizeFeatures( # Features to normalize in the graphs
        ["edge_dist", "angle_cos"]
    )]
    datamodule = CEGANNDataModule(
        # root="data/mp-data",
        # datasets="mp",
        root="data/custom-data",
        datasets="custom",
        pretreat=False,
        transforms=transforms,
        num_workers=31, # Number of CPUs to allow for data loading
        k_neigh=k_neigh
    )
    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()
    val_loader = datamodule.val_dataloader()

    # Model
    gbf_bond =  {"start": 0., "stop": 1., "num_radial": num_radial}
    gbf_angle = {"start": 0., "stop": 1., "num_radial": num_radial}
    model = CEGANN(
        gbf_bond=gbf_bond,
        gbf_angle=gbf_angle,
        n_classes=n_classes,
        edge_expansion_units=edge_expansion_units,
        angle_expansion_units=angle_expansion_units,
        n_conv_edge=n_conv_edge,
    )
    optimizer = optim.Adam
    scheduler = optim.lr_scheduler.LinearLR    
    criterion = CrossEntropyLoss(reduction='mean')
    metrics = torchmetrics.MetricCollection({
        "accuracy": torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=n_classes,
        ),
    })
    module = CEGANNModule(model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        criterion=criterion,
                        metrics=metrics)

    trainer = L.Trainer(limit_train_batches=1.0,
                        check_val_every_n_epoch=5,
                        max_epochs=10,)
    trainer.fit(model=module,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,)
    acc = trainer.test(dataloaders=test_loader,
                       ckpt_path='best')
    datamodule.teardown(stage="test")

    return acc[0]["test/accuracy"]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=48)