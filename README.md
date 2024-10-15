# Pulsar: Secure Steganography for Diffusion Models

This repository contains the code for our paper, as well as the data from our evaluation benchmarks.

This is the version of the repository submitted for consideration to the ACM CCS 2024 artifact committee. Future development will occur on the [GitHub repository](https://github.com/spacelab-ccny/pulsar), so please take a look there for updates.

## Configuration

The code for this paper was tested using the following software stack:

- Ubuntu 22.04 
- Python 3.10.14
- SageMath 10.3
- `cuda` PyTorch backend

Additional benchmarks were generated on an MacBook Pro running macOS Sonoma (`mps` PyTorch backend).

## Installation 

This installation assumes the same software setup mentioned above.

First, install the pre-requisites for figure and table generation:

```
sudo apt install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
```

Next, install the Miniforge distribution of `conda`:

```sh
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Follow the installer prompts. Press `ENTER` to view the license agreement, and then type `yes` to agree. Then, press `ENTER` to accept the default installer location.

Note: if you receive a warning about a `PYTHONPATH` variable, type `yes` at the prompt to activate `conda` on shell startup.

Next, restart your shell. If Miniforge was installed and activated correctly, you should see the following shell prompt:

```
(base) user@host:~$
```

If your prompt does not look like this, attempt the installation of Miniforge again, or make sure to run `conda init`. 

Then, we are going to [install SageMath](https://doc.sagemath.org/html/en/installation/index.html). This is simple:

```
mamba create -n sage sage python=3.10
```

Replace `3.10` with your Python version if it is different -- you can check with `python3 --version`. Make sure to press `ENTER` to confirm the installation.

After SageMath is installed, activate its environment:

```
mamba activate sage
```

You should now see the following shell prompt:

```
(sage) user@host:~$
```

If your prompt does not look like this, attempt the installation of SageMath again, or make sure to run `mamba activate sage`. 

Next, we are going to install the prerequisites for Pulsar inside of the virtual environment.

```sh
cd pulsar/  # if you're not already in this directory
pip3 install -r requirements.txt
```

Once that is done, you should be ready to run the demo and benchmarks!

## Demo

A simple demonstration of Pulsar's capabilities can be found in `demo.ipynb`, and does not require any specialized hardware to run. 

1. Open Jupyter Notebook in the Python virtual environment set up above.

```sh
jupyter notebook
```

2. Open `demo.ipynb` in the Jupyter window.
3. Click on Edit > Clear Cell Output (if you want to see everything generated from scratch).
4. Click on Run > Run All Cells.

## Benchmarks

The benchmarks from the paper can be found in `bench-results/`, and the tables and charts used in the paper can be found in `parse_benchmarks.ipynb`.

If you want to re-run the benchmarks, do the following in the Python virtual environment:

```
python3 benchmark.py
```

 Note that some of the benchmarks are configured for `cuda`, which means a recent Nvidia graphics card is required. But, the time and throughput benchmarks can run on any backend.

The runtime depends on the hardware on the system and the backend used. On an [Jetstream2 `g3.medium` instance](https://docs.jetstream-cloud.org/general/vmsizes/#jetstream2-gpu) (`cuda`), all benchmarks took about 5 hours. On a MacBook Pro with an M1 Pro chip (`mps`), only a subset of the benchmarks was run, which took about 3 hours. 

 To re-generate the tables and charts used in the paper:

1. Open Jupyter Notebook in the Python virtual environment set up above.

```sh
jupyter notebook
```

2. Open `parse_benchmarks.ipynb` in the Jupyter window.
3. Click on Edit > Clear Cell Output.
4. Click on Run > Run All Cells.

The notebook uses the latest results in generating the charts, so if you re-ran the benchmarks, you should see updated charts reflecting that new run. 

Note that both `cuda` and `mps` benchmarks need to be available for `parse_benchmarks.ipynb` to work. But, the experiments we used in our paper are included in this repository. So, if you choose to re-run the benchmarks on one device, you should still be able to run the notebook.

## Uninstallation 

The Miniforge project has instructions to [uninstall](https://github.com/conda-forge/miniforge?tab=readme-ov-file#uninstallation) the distribution, if you want to restore your regular Python environment.