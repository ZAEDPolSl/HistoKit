# Installation

We recommend installing HistoKit in a dedicated **Conda environment**.

## Quick Installation

For users familiar with Conda and pip, the installation can be summarized as follows:

```bash
git clone https://github.com/ZAEDPolSl/HistoKit.git
cd HistoKit

conda create -n histokit python=3.10 -y
conda activate histokit

python -m pip install -r requirements.txt
python -m pip install openslide-python openslide-bin

python -m pip install -e .
python -c "import histokit; print('HistoKit installed successfully!')"
```


## Prerequisites

Before installing HistoKit, make sure the following tools are available:

- **Python 3.10**
- **Anaconda or Miniconda**
- **Git**
- **pip**

## Installation from Source

### 1. Clone the Repository

Clone the HistoKit repository using Git:

```bash
git clone https://github.com/ZAEDPolSl/HistoKit.git
cd HistoKit
```

Alternatively, download the repository as a ZIP archive, extract it, and navigate to the extracted project directory.

### 2. Create a Conda Environment

Create a dedicated environment with Python 3.10:

```bash
conda create -n histokit python=3.10 -y
```

Activate the environment:

```bash
conda activate histokit
```

### 3. Install HistoKit with PIP

Install requirements:

```bash
python -m pip install -r requirements.txt
```

Navigate to the root directory of the HistoKit repository, where `pyproject.toml` or `setup.py` is located.
Install HistoKit using:

```bash
python -m pip install -e .
```

This command installs HistoKit in **editable mode**, allowing Python to import the package directly from the local source directory.

If the package metadata declares all required dependencies, pip will install them automatically.

### 5. Install OpenSlide

HistoKit uses OpenSlide for reading whole-slide images.

On supported Windows environments, install the OpenSlide Python bindings and native binaries:

```bash
python -m pip install openslide-python openslide-bin
```

### 6. Verify the Installation

Verify that HistoKit can be imported successfully:

```bash
python -c "import histokit; print('HistoKit installed successfully!')"
```

Check the installed package information:

```bash
python -m pip show histokit
```

## Troubleshooting

### HistoKit Cannot Be Imported

If you encounter:

```text
ModuleNotFoundError: No module named 'histokit'
```

Make sure the correct Conda environment is active:

```bash
conda activate histokit
```

Then reinstall HistoKit from the repository root:

```bash
python -m pip install -e .
```

### OpenSlide Errors

If OpenSlide cannot load WSI files, verify the installation:

```bash
python -m pip show openslide-python openslide-bin
```

Make sure the required native OpenSlide libraries are available for your operating system.

### Dependency Conflicts

Check for dependency conflicts:

```bash
python -m pip check
```

If dependency conflicts persist, consider creating a fresh Conda environment and reinstalling HistoKit.


