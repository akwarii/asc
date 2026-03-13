# Lightning-CEGANNv2: A Modular CEGANN Implementation with Optimized Architecture and Data Handling

[![python](https://img.shields.io/badge/Python_3.11+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai/docs/pytorch/stable/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![license](https://img.shields.io/badge/License-GNU_GPLv3-green.svg?labelColor=gray)](https://github.com/akwarii/Lightning-CEGANN2#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/akwarii/Lightning-CEGANN2/pulls)

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

## Installation

### Prerequisites

- [Python](https://www.python.org/) 3.11 and newer

### uv (recommended)

```bash
# If uv is not installed on your machine
pip install uv

# Install the virtual environment
uv sync

# Add extra options if you want to download datasets and/or use hyperparameter optimization
uv sync --group api --group hpo

# Activate the virtual environment
source .venv/bin/activate
```

### Pip

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

### Get your free API keys (optional)

**Materials Project**: If you intend to use the Materials Project dataset, get your API key [here](https://next-gen.materialsproject.org/api#api-key)

**Crystal Space Group**: To use our preprocessed dataset build upon Materials Project, AFLOW and GNoME databases, you need Kaggle account

1. Create an account on the [Kaggle website](https://www.kaggle.com/)
2. Go to your [User profile](https://www.kaggle.com/settings/account) and click on `Create New Token`
3. Move the downloaded `kaggle.json` file to the `$HOME/.kaggle` folder (if needed, create it with `mkdir $HOME/.kaggle`)

Once you got your credentials, create a `.env` file (using `cp .env.example .env` for example) and enter your credentials. Remember to not share this file with others.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

To train the PaiNN model on the Si dataset, you can run:

```bash
python fit --config configs/painn.yaml
```

The provided configuration file will train a model in the exact same way as what is presented in `examples/silicon/silicon.ipynb`

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->

## Roadmap

**Code and Framework:**

- [ ] Packaging
- [x] Production ready training script + notebook
- [x] Production ready inference script + notebook
- [ ] OVITO interface
- [x] The DataModule class must be able to accept a path as input (in addition to the dataset name) in case the user uses a custom path (for example, data/to_predict instead of data/custom)
- [ ] Update documentation
- [x] Add Silicium example
- [x] Update Node dropout
- [x] Update Edge dropout
- [ ] Ideally, replace the inference function by LightningModule.predict + LightningDataModule
- [ ] Complete all TODOs
- [ ] Find fancy name
- [ ] Choose final version of CSG dataset

**Optional:**

- [ ] Graph coarsening transformation
- [ ] Allows to use graph embedding for clustering

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
