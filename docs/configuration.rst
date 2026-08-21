Configuration
=============

The project does not use a single monolithic YAML file. Instead, the training command loads a small set of configuration fragments and merges them together at runtime. The main CLI command is typically:

.. code-block:: bash

   python main.py fit \
     --config configs/main.yaml \
     --trainer configs/trainer/default.yaml \
     --model configs/model/painn.yaml \
     --data configs/data/custom.yaml

This makes it easy to swap models, datasets, or training settings without editing the entire configuration.

Main configuration
------------------

The main configuration file sets global runtime options and is stored in ``configs/main.yaml``.

.. code-block:: yaml

   seed_everything: 3
   ckpt_path: null
   weights_only: null

These values control the random seed and optional checkpoint loading behavior.

Model configuration
-------------------

The model configuration file defines the optimizer, training loop settings, and the model object itself. In this project, the default model file is ``configs/model/painn.yaml``.

.. code-block:: yaml

   lr: 0.006
   warmup: 700
   max_iters: -1  # Will be calculated automatically in main.py
   compile: true

   model:
     class_path: src.models.PaiNN
     init_args:
       hidden_channels: 32
       num_layers: 2
       num_radial: 4

   metrics:
     - class_path: torchmetrics.F1Score
     - class_path: torchmetrics.AUROC
     - class_path: torchmetrics.Accuracy
     - class_path: torchmetrics.ConfusionMatrix

Key points:

* ``lr`` is the learning rate.
* ``warmup`` is the number of steps used before full optimization.
* ``model.class_path`` points to the model implementation to instantiate.
* ``model.init_args`` contains the constructor arguments for that model.
* ``metrics`` is a list of metric classes that will be tracked during training.

Data configuration
------------------

The dataset configuration defines which dataset to use and how it should be split and transformed. An example is ``configs/data/custom.yaml``.

.. code-block:: yaml

   dataset_name: custom
   root: examples/silicon/data
   lengths: [0.7, 0.2, 0.1]
   batch_size: 128
   num_workers: 8
   use_imbalance_sampler: true

   transforms:
     - class_path: src.transforms.RandomPerturbation
       init_args:
         std_range: [0.0, 0.05]
     - class_path: src.transforms.BoxStrain
       init_args:
         std_range: [0.0, 0.05]
         directions: all

This file is responsible for:

* selecting the dataset via ``dataset_name``;
* pointing to the data directory through ``root``;
* splitting the dataset into train/validation/test sets with ``lengths``;
* controlling batching and parallel data loading;
* applying training-time augmentations via the ``transforms`` list.

Trainer configuration
---------------------

The trainer configuration controls Lightning's runtime behavior, logger, and callbacks. The default example is ``configs/trainer/default.yaml``.

.. code-block:: yaml

   accelerator: auto
   devices: auto
   max_epochs: 50
   precision: bf16-mixed
   enable_model_summary: false # we replace the default model summary with a custom one in the callbacks

   logger:
     class_path: lightning.pytorch.loggers.TensorBoardLogger
     init_args:
       save_dir: logs
       name: painn_experiment
       default_hp_metric: false

   callbacks:
     - class_path: lightning.pytorch.callbacks.ModelCheckpoint
       init_args:
         monitor: val/loss
         mode: min
         save_top_k: 3
         filename: "{epoch}-{val_loss:.4f}"
     - class_path: lightning.pytorch.callbacks.EarlyStopping
       init_args:
         monitor: val/loss
         patience: 20
         mode: min
     - class_path: lightning.pytorch.callbacks.RichModelSummary
       init_args:
         max_depth: 2

This file configures:

* the hardware target through ``accelerator`` and ``devices``;
* the number of epochs and numerical precision;
* the TensorBoard logger;
* checkpointing and early stopping callbacks.

Configuration pattern
---------------------

The YAML format follows the ``class_path`` / ``init_args`` pattern used by `LightningCLI` and `jsonargparse`. This allows complex objects such as models, loggers, and callbacks to be constructed directly from configuration files without hard-coding them in Python.

In practice, this means each configuration file is a partial specification rather than an isolated script. The CLI merges them into a single runtime configuration before launching training.
