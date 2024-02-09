# Lightning-CEGANNv2: A Modular CEGANN Implementation with Optimized Architecture and Data Handling

[![python](https://img.shields.io/badge/-Python_3.10_%7C_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) <br>

<!-- TODO Need to add GitHub Actions test -->
[![tests](https://github.com/akwarii/Lightning-CEGANN2/actions/workflows/test.yml/badge.svg)](https://github.com/akwarii/Lightning-CEGANN2/actions/workflows/test.yml)
<!-- TODO Need to add GitHub Actions code quality -->
[![code-quality](https://github.com/ashleve/lightning-hydra-template/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/ashleve/lightning-hydra-template/actions/workflows/code-quality-main.yaml)
[![codecov](https://codecov.io/gh/ashleve/lightning-hydra-template/branch/main/graph/badge.svg)](https://codecov.io/gh/ashleve/lightning-hydra-template) <br>

[![license](https://img.shields.io/badge/License-GNU_GPLv3-green.svg?labelColor=gray)](https://github.com/akwarii/Lightning-CEGANN2#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/akwarii/Lightning-CEGANN2/pulls)
[![contributors](https://img.shields.io/github/contributors/ashleve/lightning-hydra-template.svg)](https://github.com/akwarii/Lightning-CEGANN2/contributors)

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

<!-- GETTING STARTED -->
<!-- TODO Write Getting Started -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```
### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
<!-- TODO Write Examples -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

**Code and Framework:**

- [X] Extensive code refactoring for readability, maintainability, and future development.
- [X] Transition from Ignite to Lightning:
    - [X] LightningModule for structured training.
    - [ ] DataModule for data loading and preprocessing.
    - [ ] Trainer for optimal hyperparameter tuning and training execution.

**Model Optimizations:**

- [X] Adoption of GATv2 attention mechanism.
- [X] Replacement of LayerNorm with GraphNorm for enhanced graph data handling.
- [ ] Planned RBF optimization for potentially improved performance.
- [ ] Integration of Angular Basis Function inspired by GemNet for potentially improved feature representation.

**Data Handling:**

- [X] Leveraged existing Material Project and Aflow datasets with custom Aflow API wrapper.
- [ ] Support for user-defined structures readable by pymatgen.
- [ ] Exploration of data augmentation techniques:
    - [ ] Generic Structure Augmentation class
    - [ ] Incorporation of Gaussian noise and MD simulations (with ASE).
    - [ ] Potential investigation of active learning for improved training efficiency.

**Graph Creation Optimization:**
- [X] Improvement of the feature computation speed.
- [X] Efficient KNN graph construction.
- [ ] Exploration of radius graphs.
- [ ] Flexible graph factory for better maintainability.

**Additional Enhancements:**

- [ ] Integration of Optuna for efficient hyperparameter tuning and model pruning.
- [ ] Implementation of a comprehensive logging system (Neptune/TensorBoard/Wandb) for detailed analysis and reproducibility.
- [ ] Creation of a user-friendly command-line interface (CLI) based on argparse for ease of use.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- EXPERIMENTS -->
<!-- TODO This should be removed in the future -->
## Experiments

* Planned experiment: Local classification of individual space groups within the material.
* More complex experiment: Simultaneous classification of both space groups and grain boundaries within the material.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Experimental Guidelines

1. Begin with space groups relevant to the target material (e.g., zirconia).
2. Introduce grain boundaries into the data repository for the combined experiment.
3. Aim to classify all 230 space groups, prioritizing well-converged structures from the Material Project.
4. Extend to grain boundary classification if initial space group classification proves successful.

This research presents Lightning-CEGANN2 as a modular and optimized implementation with novel enhancements to the architecture and data handling pipeline. Future work will focus on completing planned developments, conducting comprehensive experiments, and evaluating the effectiveness of the proposed improvements.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- HOW TO CITE -->
<!-- TODO Add citation -->
## How to cite
If you use ``Lightning-CEGANNv2`` in your research, please consider citing the following work:

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

If you have any questions, please contact one of the contributors below:<br/>
&emsp; Gaël Huynh - <gael.huynh@univ-lyon1.fr>

If you found a bug or want to request a new feature, please create a new 
[GitHub Issues](https://github.com/akwarii/Lightning-CEGANN2/issues)

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License
Distributed under the GNU GPLv3 License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
