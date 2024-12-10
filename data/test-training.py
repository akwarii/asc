# Based on
# https://lightning.ai/docs/pytorch/stable/starter/introduction.html

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

n_classes = 230 # # placeholder, counted by hand
# n_classes = 8

# Data management
transforms=[NormalizeFeatures( # Features to normalize in the graphs
    ["edge_dist", "angle_cos"]
)]
datamodule = CEGANNDataModule(
    # root="data/mp-data", datasets="mp", # Works !
    # root="data/aflow-data", datasets="aflow", # Works ?
    # root="data/csg-data", datasets="csg", # Dataset doesn't exist
    root="data/gnome-data", datasets="gnome", # Works !
    # root="data/custom-data", datasets="custom", # test this (needs the following line also)
    # pretreat=True, origin_dir="/home/dbissuel/Documents/softs/sourcecodes/CEGANN/pretrained/spacegroup/",
    # pretreat=False,
    transforms=transforms,
    num_workers=31, # Number of CPUs to allow for data loading
    k_neigh=12,
    # download=True,
    train_val_test_split=[0.5, 0.25, 0.25],
)
datamodule.setup(stage="fit")
train_loader = datamodule.train_dataloader()
test_loader = datamodule.test_dataloader()
val_loader = datamodule.val_dataloader()

# Model
gbf_bond = {
    "start": 0.,
    "stop": 1.,
    "num_radial": 80 # DB
}
gbf_angle = {
    "start": 0.,
    "stop": 1.,
    "num_radial": 80 # DB
}
model = CEGANN(
    gbf_bond=gbf_bond,
    gbf_angle=gbf_angle,
    n_classes=n_classes,
    edge_expansion_units=256, # OG paper 256
    angle_expansion_units=256,# OG paper 256
)
optimizer = optim.Adam
# optimizer = optim.SGD
# scheduler = optim.lr_scheduler.LinearLR
scheduler = optim.lr_scheduler.StepLR
# scheduler = optim.lr_scheduler.ExponentialLR
scheduler_params = {"gamma": 0.5, "step_size": 100}
criterion = CrossEntropyLoss(reduction='mean')
metrics = torchmetrics.MetricCollection(
    {
        "accuracy": torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=n_classes,
        ),
    }
)
module = CEGANNModule(model=model,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      scheduler_params=scheduler_params,
                      criterion=criterion,
                      metrics=metrics)

# Training
trainer = L.Trainer(limit_train_batches=1.0,
                    check_val_every_n_epoch=5,
                    max_epochs=200,)
trainer.fit(model=module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,)
out = trainer.test(dataloaders=test_loader,
                   ckpt_path='best')
datamodule.teardown(stage="test")
print("After testing :", out)

# Prediction
# datamodule.setup(stage="predict")
# pred_loader = datamodule.predict_dataloader()
# acc = trainer.predict(model=module,
#                       dataloaders=pred_loader,
#                       return_predictions=True,
#                       ckpt_path='best',)
# datamodule.teardown(stage="predict")