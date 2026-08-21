# Atomic Structure Classification

[![python](https://img.shields.io/badge/Python_3.11+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai/docs/pytorch/stable/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![license](https://img.shields.io/badge/License-GNU_GPLv3-green.svg?labelColor=gray)](https://github.com/akwarii/asc#license)
[![PRs](https://img.shields.io/badge/PRs-welcomed-brightgreen.svg)](https://github.com/akwarii/asc/pulls)

<!-- TABLE OF CONTENTS -->

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#main-technologies">Main Technologies</a></li>
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
    <li><a href="#how-to-cite">How to cite</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

ASC (Atomic Structure Classification, pronunced ask) is a high-performance framework designed to accelerate the development and benchmarking of Graph Neural Networks for atomistic simulations.

By decoupling the data engineering from the model design, this library allows researchers to ignore the data pipeline barrier and focus entirely on model innovation. An [OVITO extension](https://github.com/akwarii/asc-extension) is also provided to allow researchers to quickly load trained models directly into OVITO to perform real-time inference and assess performances visually.
</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Main Technologies -->

## Main Technologies

[PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) - a PyTorch wrapper made to avoid boilerplate code and improve reproductibility, allowing fast iterations.

[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) - a flexible library build upon PyTorch to easily handle Graph Neural Networks.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Installation

### Prerequisites

- [Python](https://www.python.org/) 3.11 or newer

### Installation

We recommend using `uv` to install the library dependencies:

```bash
# Clone the repo
git clone https://github.com/akwarii/asc.git
cd asc

# If uv is not installed on your machine
pip install uv

# Install the virtual environment and dependencies
uv sync

# Add extra options if you want to download datasets and/or use hyperparameter optimization
uv sync --group api --group hpo
uv pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-${TORCH}+${CUDA}.html

# Activate the virtual environment
source .venv/bin/activate
```

where

- ${TORCH} should be replaced by either 2.8.0, 2.9.0, or 2.10.0
- ${CUDA} should be replaced by either cpu, cu126, cu128, cu129, or cu130

Alternatively, you can use pip directly:

```bash
# Clone the repo
git clone https://github.com/akwarii/asc.git
cd asc

# (OPTIONAL) Create a conda environment
conda create -n asc python=3.11
conda activate asc

# Install the requirements
pip install -r requirements.txt
pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-${TORCH}+${CUDA}.html
```

### Get your free API keys (optional)

**Materials Project**: If you intend to use the Materials Project dataset, get your API key [here](https://next-gen.materialsproject.org/api#api-key)

**Crystal Space Group**: To use our preprocessed dataset build upon Materials Project, AFLOW and GNoME databases, you need a Kaggle account.

1. Create an account on the [Kaggle website](https://www.kaggle.com/)
2. Go to your [User profile](https://www.kaggle.com/settings/account) and click on `Create New Token`
3. Move the downloaded `kaggle.json` file to the `$HOME/.kaggle` folder (if needed, create it with `mkdir $HOME/.kaggle`)

Once you got your credentials, create a `.env` file (using `cp .env.example .env` for example) and enter your credentials. Remember to not share this file with others.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

<!-- TODO add more details? -->

To train the PaiNN model on the Si (silicon) dataset, you can run:

```bash
python main.py fit --config configs/main.yaml --trainer configs/trainer/default.yaml --model configs/model/painn.yaml --data configs/data/custom.yaml
```

The provided configuration file will train a model in the same way as what is presented in `examples/silicon/silicon.ipynb`.

`main.py` script can also be used to run validation/test steps as well as inference. The available subcommands are `fit`, `validate`, `test`, and `predict`. More details can be obtained by running

```bash
python main.py --help
```

The CLI can give you more information about a specific argument by running:

```bash
python main.py fit --model.model.help src.models.PaiNN
python main.py fit --data.dataset.help
```

The first will print the help message related to our implementation of the PaiNN model, while the other will display the arguments that can be passed to all our datasets.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->

## Roadmap

**Code and Framework:**

- [ ] Config file for each dataset and model
- [ ] Document how to extend the code
- [ ] Node-level datamodule
- [ ] Improve documentation
- [ ] Switch between different CSG dataset version
- [ ] Make real tests

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Documentation

In order to install the dependencies required for building the documentation, run the following command:

```bash
uv sync --group doc
```

You can then build the documentation by running:

```bash
make docs
```

From there, you can open the `docs/_build/html/index.html` file in your browser to view the documentation.

<!-- CONTRIBUTING -->

## Contributing

<!-- TODO Add more details to the contribution guideline -->

Follow the generic coding conventions defined in [PEP8](https://peps.python.org/pep-0008/).
Run pre-commit before submitting a PR by running `make format`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- TODO Add article ref once published -->

## How to cite

If you use `asc` in your research, please consider citing the following work:

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

If you have any questions, please contact one of the contributors below:

<ul>
  <li>
    <a href="mailto:gael.huynh@gmail.com?"> Gaël Huynh</a>
  </li>
  <li>
    <a href="mailto:dylan.bissuel@univ-lyon1.fr?"> Dylan Bissuel</a>
  </li>
  <li>
    <a href="mailto:david.rodney@univ-lyon1.fr?"> David Rodney</a>
  </li>
</ul>

If you found a bug or want to request a new feature, please create a new
[GitHub Issues](https://github.com/akwarii/asc/issues)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
