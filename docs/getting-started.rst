Getting started
===============

This section provides instructions for installing the project, building the documentation, and running an example.

Installing the project
----------------------

Preambles
^^^^^^^^^

This project requires Python 3.11 or higher. We recommend using the `uv` tool to create and manage virtual environments dedicated to this project. For more information, see the `uv documentation <https://docs.astral.sh/uv/getting-started/installation/>`_.


The project relies on the following core libraries for deep learning and graph neural networks:

* `PyTorch Lightning <https://lightning.ai/>`_,  a PyTorch wrapper made to avoid boilerplate code and improve reproductibility, allowing fast iterations.
* `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_, a flexible library build upon PyTorch to easily handle Graph Neural Networks.

It also uses the additional libraries for data processing and visualization. You can find the complete list of dependencies in the ``pyproject.toml`` file.

With `uv`, you can install the project and its dependencies in a virtual environment by running the following command from the project root:

.. code-block:: bash

   uv sync

This automatically creates a virtual environment with a compatible Python version and installs the project along with its dependencies. You can then activate the virtual environment by running:

.. code-block:: bash

   source .venv/bin/activate

and deactivate it by running:

.. code-block:: bash

   deactivate

once you are done working with the project.

We also provide out-of-the-box dependency groups for specific tasks.

Installing with datasets APIs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install the project with the datasets APIs, run:

.. code-block:: bash

   uv sync --group api

Hyperparameter optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install the project with hyperparameter optimization dependencies, run:

.. code-block:: bash

   uv sync --group hpo

Notebook support
^^^^^^^^^^^^^^^^

To install the project with notebook support, run:

.. code-block:: bash

   uv sync --group notebook

Building the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Install the project with its documentation dependencies:

.. code-block:: bash

   uv sync --group doc

Build this site from the repository root:

.. code-block:: bash

   make docs

Open ``docs/_build/html/index.html`` in a browser.

Installing several dependency groups
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to install several dependency groups, you can chain them in the command line. For example, to install the project with both the datasets APIs and hyperparameter optimization dependencies, run:

.. code-block:: bash

   uv sync --group api --group hpo

Inference on large graphs
^^^^^^^^^^^^^^^^^^^^^^^^^

To accelerate inference on large graphs, we use a last external dependency, `PyG lib <https://github.com/pyg-team/pyg-lib>`_. This provides low-level graph neural network operators for PyTorch Geometric, allowing for more efficient computations that are particularly useful when working large systems.

As it is not currently available on the official PyPI repository, we have to specify a custom URL, depending on the PyTorch and CUDA versions you are using.

The following command installs the library for the current PyTorch and CUDA versions:

.. code-block:: bash

   uv pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-${TORCH}+${CUDA}.html

where

* ``${TORCH}`` should be replaced by either 2.8.0, 2.9.0, or 2.10.0, depending on the PyTorch version you are using.
* ``${CUDA}`` should be replaced by either cu126, cu128, cu129, or cu130, depending on the CUDA version you are using, or by cpu if you are not using CUDA.

Running an example
------------------
To train the provided silicon example, run:

.. code-block:: bash

   python main.py fit --config configs/main.yaml --trainer configs/trainer/default.yaml --model configs/model/painn.yaml --data configs/data/custom.yaml

This command can be broken down as follows:

* ``python main.py fit``: runs the training loop.
* ``--config configs/main.yaml``: specifies the main configuration file.
* ``--trainer configs/trainer/default.yaml``: specifies the trainer configuration file.
* ``--model configs/model/painn.yaml``: specifies the model configuration file.
* ``--data configs/data/custom.yaml``: specifies the data configuration file.

Each of these configuration files are explained in detail in the `configuration section <configuration.html>`_ of the documentation.