# Lightning-CEGANNv2: A Modular CEGANN Implementation with Optimized Architecture and Data Handling

[![python](https://img.shields.io/badge/-Python_3.10_%7C_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) <br>
[![tests](https://github.com/akwarii/Lightning-CEGANN2/actions/workflows/test.yaml/badge.svg)](https://github.com/akwarii/Lightning-CEGANN2/actions/workflows/test.yaml)
[![code-quality](https://github.com/akwarii/Lightning-CEGANN2/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/akwarii/Lightning-CEGANN2/actions/workflows/code-quality-main.yaml)
[![codecov](https://codecov.io/gh/akwarii/Lightning-CEGANN2/branch/main/graph/badge.svg)](https://codecov.io/gh/akwarii/Lightning-CEGANN2) <br>
[![license](https://img.shields.io/badge/License-GNU_GPLv3-green.svg?labelColor=gray)](https://github.com/akwarii/Lightning-CEGANN2#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/akwarii/Lightning-CEGANN2/pulls)
[![contributors](https://img.shields.io/github/contributors/akwarii/Lightning-CEGANN2.svg)](https://github.com/akwarii/Lightning-CEGANN2/contributors)

<!-- TABLE OF CONTENTS -->

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

This work introduces Lightning-CEGANNv2, a novel implementation of a modified Crystal Edge Graph Attention Neural Network (CEGANN) leveraging the modularity and efficiency of the Lightning framework. This research aims to enhance the original CEGANN architecture and data handling pipeline for improved performance and flexibility. The original CEGANN architecture can be found here: <a href="https://www.nature.com/articles/s41524-023-00975-z">CEGANN: Crystal Edge Graph Attention Neural Network for multiscale classification of materials environment</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- EXPERIMENTS -->

<!-- TODO This should be removed in the future -->

## Experiments

- Local classification of individual space groups within the material.
- Simultaneous classification of both space groups and grain boundaries within the material.

### Experimental Guidelines

1. Begin with space groups relevant to the target material (e.g., zirconia).
2. Introduce grain boundaries into the data repository for the combined experiment.
3. Aim to classify all 230 space groups, prioritizing well-converged structures from the Material Project.
4. Extend to grain boundary classification if initial space group classification proves successful.

This research presents Lightning-CEGANN2 as a modular and optimized implementation with novel enhancements to the architecture and data handling pipeline. Future work will focus on completing planned developments, conducting comprehensive experiments, and evaluating the effectiveness of the proposed improvements.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Main Technologies -->

## Main Technologies

[PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) - a lightweight PyTorch wrapper made to avoid boilerplate code and improve reproductibility.

[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) - a flexible library build upon PyTorch to easily handle Graph Neural Networks.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

### Prerequisites

- [Python](https://www.python.org/) 3.8 and newer
- (OPTIONAL) [Ovito](https://www.ovito.org/) (to visualize the classification results)

### Installation

#### Pip

```bash
# Clone the repo
git clone https://github.com/akwarii/Lightning-CEGANN2.git
cd Lightning-CEGANN2

# (OPTIONAL) Create a conda environment
conda create -n cegann
conda activate cegann

# Install the requirements
pip install -r requirements.txt
```

### Conda

*The environment.yaml file is not ready to use, please use pip*

```bash
# Clone the repo
git clone https://github.com/akwarii/Lightning-CEGANN2.git
cd Lightning-CEGANN2

# (OPTIONAL) Create a conda environment
conda create -f environment.yaml -n cegann
conda activate cegann
```

### Get your free API keys (optional)

**Materials Project**: If you intend to use the Materials Project dataset, get your API key [here](https://next-gen.materialsproject.org/api#api-key)

**Crystal Space Group**: To use our preprocessed dataset build upon Materials Project, AFLOW and GNoME databases, you need Kaggle account

1. Create an account on the [Kaggle website](https://www.kaggle.com/)
2. Go to your [User profile](https://www.kaggle.com/settings/account) and click on `Create New Token`
3. Move the downloaded `kaggle.json` file to the `$HOME/.kaggle` folder (if needed, create it with `mkdir $HOME/.kaggle`)

Once you got your credentials, create a `.env` file (using `cp .env.example .env` for example) and enter your credentials. Remember to not share this file with others.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

<!-- TODO Write Examples -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Project Structure -->

## Project structure

The project is currently organized as follow:

```
├── .github                       <- Github Actions workflow
│
├── configs                       <- Configs to use to train / test models
│
├── data                          <- Project data (downloaded data will be here)
│
├── logs                          <- Logs generated by Lightning loggers
│
├── notebooks                     <- Examples notebooks can be found here
│
├── old                           <- TO REMOVE, just for reference
│
├── src                           <- Source code
│   ├── api                           <- AFLOW API wrapper
│   │
│   ├── data                          <- Data scripts
│   │   ├── augmentation                  <- Data augmentation (after graph transformations)
│   │   ├── datasets                      <- Datasets definition
│   │   ├── pyg_data                      <- TEMPORARY FOLDER, work in progress to handle PyG data with Lightning
│   │   ├── sampler                       <- Node sampling methods
│   │   ├── transforms                    <- Transformation to apply to the graphs (before augmentation)
│   │   │
│   │   └── cegann_datamodule.py          <- Contains the LightningDataModule
│   │
│   ├── models                        <- Model scripts
│   │   ├── components                    <-
│   │   │   ├── expansion                     <- Basis expansion blocks
│   │   │   ├── layers                        <- Building blocks of the models
│   │   │   │
│   │   │   └── cegann.py                     <- CEGANN model definition
│   │   │
│   │   └── cegann_module.py            <- Contains the LightningModule
│   │
│   ├── processing                    <- Graph processing
│   │
│   ├── utils                         <- Utilities scripts such (eg constants definition)
│   │
│   ├── eval.py                       <- Handles model evaluation
│   └── train.py                      <- Handles model training
│
├── tests                         <- Unit tests
│
├── .env.example                  <- Example of file for storing API credentials
├── .gitignore                    <- List of files ignored by git
├── .pre-commit-config.yaml       <- pre-commit hooks for code formatting
├── environment.yml               <- File to install the conda environment
├── LICENSE                       <- License file
├── Makefile                      <- Makefile with useful command shortcuts
├── requirements.txt              <- File to install python dependencies
└── README.md
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->

## Roadmap

**Code and Framework:**

- [x] Extensive code refactoring for readability, maintainability, and future development.
- [x] Transition from Ignite to Lightning:
  - [x] LightningModule for structured training.
  - [x] DataModule for data loading and preprocessing.
  - [ ] Trainer for optimal hyperparameter tuning and training execution.
- [ ] Transition from pure PyTorch to PyG
  - [x] Sparse graphs
  - [ ] Neighbour loader to enable inference on large graphs
  - [ ] Dynamic batch sampling to train the model on graphs with variable number of nodes without OOM
  - [ ] Change existing modules to PyG MessagePassing subclass
- [ ] Model compilation

**Model Optimizations:**

- [x] GATv2 attention mechanism
- [ ] Looking for a better normalization than LayerNorm
- [x] Radial basis
  - [x] Gaussian expansion
  - [x] Circular Bessel expansion
- [x] Angular Basis Function inspired by eg. GemNet and DimeNet
  - [x] Gaussian expansion
  - [x] Spherical harmonics (m=0)
- [x] Envelope for smooth cutoff, user can to disable it
  - [x] Exponential
  - [x] Polynomial

**Datasets**

- [x] Aflow data with a custom API wrapper
- [x] Material Project data
- [x] GNoMe summary dataset
- [x] Cleaning and merging the mentioned datasets and upload it to Kaggle
- [x] User-defined dataset for structures readable by pymatgen

**Data augmentation/transformations**

- [x] Data transformations techniques:
  - [x] Normalization transformation
  - [ ] Tags to remove atoms from structure (pre-transformation)
- [ ] Data augmentation techniques:
  - [ ] Incorporation of Gaussian noise (with ASE)
  - [ ] Incorporation of MD simulations (with ASE)
  - [ ] Drop random nodes to simulate defects.

**Graph Creation Optimization:**

- [x] Improvement of the feature computation speed.
- [x] Efficient KNN graph construction.
- [x] Flexible graph factory for better maintainability.
- [x] Create the graphs on-the-fly to avoid large memory usage.
- [x] Collate and save the whole graph dataset to load it later.

**Additional Enhancements:**

- [ ] Integration of Optuna for efficient hyperparameter tuning and model pruning.
- [ ] Implementation of at least one comprehensive logging system (Neptune/TensorBoard/Wandb) for detailed analysis and reproducibility.
- [ ] Use of a config file to make use of e.g., Hydra + Submitit

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->

<!-- TODO Add more details to the contribution guideline -->

## Contributing

Follow the generic coding conventions defined in [PEP8](https://peps.python.org/pep-0008/).
Run pre-commit before submitting a PR by running `make format`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- HOW TO CITE -->

<!-- TODO Add reference -->

## How to cite

If you use `Lightning-CEGANNv2` in your research, please consider citing the following work:

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the GNU GPLv3 License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

If you have any questions, please contact one of the contributors below:

<ul>
  <li>
    <a href="mailto:gael.huynh@univ-lyon1.fr?"> Gaël Huynh</a>
  </li>
  <li>
    <a href="mailto:david.rodney@univ-lyon1.fr?"> David Rodney</a>
  </li>
</ul>

If you found a bug or want to request a new feature, please create a new
[GitHub Issues](https://github.com/akwarii/Lightning-CEGANN2/issues)

Project Link: [https://github.com/akwarii/Lightning-CEGANN2](https://github.com/akwarii/Lightning-CEGANN2)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
