# How to install `faiss-gpu` for unsupported GPU architectures

The `faiss-gpu-cu12` package used in `pyproject.toml` is based on [faiss-wheels](https://github.com/faiss-wheels/faiss-wheels) which does not provide precompiled wheels for all GPU architectures.

As of writing this document (October 2025), I have managed to use it on :

- a local machine with an NVIDIA RTX 4000 Ada Generation GPU.
- the IN2P3 computing center using NVIDIA V100 GPUs.

For NVIDIA H100 GPUs, it seems the no fitting wheel can be made available due to Pypi restrictions on the maximum wheel size.

Therefore, the easiest solution is to build the wheel locally before using it in the project.

## Building the wheel locally

1. Clone the `faiss-wheels` repository:

   ```bash
   git clone https://github.com/faiss-wheels/faiss-wheels.git
   ```

2. Navigate to the `faiss-wheels` directory:

   ```bash
   cd faiss-wheels
   ```

3. Edit the wheel to change the name of the package (default is `faiss-cpu`)

   ```bash
   sed -i 's/faiss-cpu/faiss-gpu/g' pyproject.toml
   ```

4. ***(optional depending on your machine)*** ensure you have access to a GPU and CUDA installed. You can check this by running:

   ```bash
   nvidia-smi
   ```

   On computing facilities, it might require you to connect to a GPU node, for example using `srun` on SLURM (note: the node needs to have internet access).

   Alternatively, yo can load the CUDA module if available.

5. Build the wheel using the following command (you can specify the CUDA version if needed):

   ```bash
   export FAISS_OPT_LEVELS=generic,avx2,avx512  # Optional: customize for your machine
   export FAISS_GPU_SUPPORT=CUDA  # Enable GPU support   
   uv build --wheel  # Build the wheel, it might take a while
   ```

6. After the build is complete, you should find the wheel file in the `dist` directory.

## Using the locally built wheel in your project

1. Navigate to your project directory where the `pyproject.toml` file is located.

2. Install the rest of the dependencies using `uv`:

   ```bash
   uv venv [venv_name]  # venv_name is optional
   source [venv_name]/bin/activate  # Activate the virtual environment
   uv sync [options] --active # use --active if you have several environments to specify the active one
   ```

3. Install the locally built `faiss-gpu` wheel using `pip`:

   ```bash
   uv pip install /path/to/faiss-wheels/dist/faiss_gpu-*.whl
   ```

4. Verify the installation by running a Python shell and importing `faiss`:

   ```python
   import faiss

   print(faiss.__version__)
   print(hasattr(faiss, "StandardGpuResources"))  # Should return True if GPU support is enabled
   ```

5. You can run the `tests/test_benchmark_knn.py` script to verify that `faiss-gpu` is working correctly with your GPU.
