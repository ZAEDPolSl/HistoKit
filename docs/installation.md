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

## External models

### GrandQC

GrandQC provides deep learning-based artifact segmentation for whole-slide images (WSIs).

!!! warning "License and citation requirements"

    GrandQC is available for academic research and non-commercial use only.
    See [license.md](./license.md) for licensing details.

    If you use GrandQC, please cite the original publication:

    [:material-format-quote-close: Weng et al. (2024). GrandQC: A comprehensive solution to quality control problem in digital pathology. Nature Communications, 15, 10685.](https://doi.org/10.1038/s41467-024-54769-y)

#### Model download

To use GrandQC-based artifact segmentation, download the pretrained model weights from the external repository:

[:material-download: Download GrandQC models from Zenodo](https://zenodo.org/records/14041538)

The following model resolutions are available:

| Resolution (µm/px) | Magnification |
| :----------------- | :------------ |
| MPP 1.0            | ~10×          |
| MPP 1.5            | ~7.5×         |
| MPP 2.0            | ~5×           |

!!! tip "Model selection"

    According to the article, higher magnification (lower MPP)
    provides more precise artifact segmentation, while lower magnification
    offers faster processing with a minor reduction in artifact segmentation quality.

#### Configuration

After downloading the selected model, update the `model_path` parameter
in the `config/grandqc.yaml` configuration file to point to the downloaded
model weights.

```yaml title="config/grandqc.yaml" hl_lines="7"
# output
saver: hdf5
out_dir: ./outputs/
save_mag: 10.0

# model
model_path: /path/to/models/GrandQC_MPP1.pth
```

Replace the example `model_path` with the actual path to your downloaded model file.


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


