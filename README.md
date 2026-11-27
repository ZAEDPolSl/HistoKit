<h1 align="center">HistoKit</h1>

<p align="center">
  <strong>A Python toolkit for computational pathology and whole-slide image processing</strong>
</p>

<p align="center">
  <a href="https://zaedpolsl.github.io/HistoKit/">
    <img src="https://img.shields.io/badge/Documentation-HistoKit-6750A4?style=flat-square&logo=materialformkdocs&logoColor=white" alt="Documentation">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://github.com/ZAEDPolSl/HistoKit/actions/workflows/docs.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/ZAEDPolSl/HistoKit/docs.yml?style=flat-square&label=docs" alt="Documentation status">
  </a>
  <a href="https://doi.org/10.5220/0000222000004070">
    <img src="https://img.shields.io/badge/DOI-10.5220%2F0000222000004070-blue?style=flat-square" alt="DOI">
  </a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="https://zaedpolsl.github.io/HistoKit/">Documentation</a> •
  <a href="#-citation">Citation</a>
</p>

---

**HistoKit** is a Python package designed to simplify whole-slide image
(WSI) processing in digital and computational pathology research.

It integrates tissue detection, artifact detection, patch extraction,
stain normalization, and image augmentation into a unified toolkit,
supporting histopathological image analysis and deep learning workflows.

<p align="center">
  <img src="https://github.com/user-attachments/assets/5987b387-7b6d-4a5e-9a8d-bff75968955e" alt="HistoKit workflow overview" width="100%">
</p>

## ✨ Features

| Feature | Description |
| :--- | :--- |
| 🔬 **Tissue detection** | Identify tissue regions within whole-slide images and separate them from the background. |
| 🧹 **Artifact detection** | Detect histological artifacts using GrandQC-based deep learning segmentation. |
| 🧩 **Patch extraction** | Extract image patches with configurable size, magnification, and stride. |
| 🎨 **Stain normalization** | Reduce staining variability using Macenko, Vahadane, and Reinhard normalization methods. |
| 🔄 **Image augmentation** | Apply common image transformations for histopathological image analysis and deep learning workflows. |

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/ZAEDPolSl/HistoKit.git
cd HistoKit
```

### 2. Create a Python environment

We recommend using a dedicated Conda environment.

```bash
conda create -n histokit python=3.10 -y
conda activate histokit
```

### 3. Install HistoKit

Install the required dependencies and the OpenSlide library:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install openslide-bin openslide-python
```

Install HistoKit as a Python package:

```bash
python -m pip install .
```

Alternatively, HistoKit can be installed from a downloaded ZIP archive.

📖 For detailed installation instructions, alternative installation
methods, and configuration options, see the
[Installation Guide](https://zaedpolsl.github.io/HistoKit/installation/).

### 🧠 Optional: GrandQC Artifact Segmentation

HistoKit integrates GrandQC for deep learning-based artifact segmentation
in whole-slide images.

This functionality requires pretrained model weights and additional
configuration.

- 📥 [Download pretrained GrandQC models from Zenodo](https://zenodo.org/records/14041538)
- ⚙️ [Model installation and configuration](https://zaedpolsl.github.io/HistoKit/installation/)
- 📄 [GrandQC paper](https://doi.org/10.1038/s41467-024-54769-y)
- 💻 [Original GrandQC repository](https://github.com/cpath-ukk/grandqc)
- 📜 [GrandQC license](https://github.com/cpath-ukk/grandqc?tab=License-1-ov-file)

GrandQC is a third-party model intended for academic, non-commercial use only.
It is distributed under its own license. Please refer to the original
repository and license for detailed terms of use.

If you use GrandQC-based artifact segmentation in HistoKit,
please also cite the original GrandQC publication:

> Weng, Z. et al. (2024).  
> **GrandQC: A comprehensive solution to quality control problem in digital pathology.**  
> Nature Communications, 15, 10685.

## 📚 Documentation

The HistoKit documentation provides detailed installation instructions,
configuration options, practical examples, tutorials, API reference,
and scientific references.

<p align="center">
  <a href="https://zaedpolsl.github.io/HistoKit/">
    <img src="https://img.shields.io/badge/Explore%20the%20Documentation-6750A4?style=for-the-badge&logo=materialformkdocs&logoColor=white" alt="Explore the Documentation">
  </a>
</p>

| Resource | Description |
| :--- | :--- |
| 🚀 [Installation Guide](https://zaedpolsl.github.io/HistoKit/installation/) | Environment setup, dependencies, and model configuration. |
| 🎓 [Tutorials and Examples](https://zaedpolsl.github.io/HistoKit/examples/) | Practical examples and image processing workflows. |
| ⚙️ [API Reference](https://zaedpolsl.github.io/HistoKit/api/) | Python modules, classes, and functions. |
| 📝 [Citations](https://zaedpolsl.github.io/HistoKit/citations/) | Scientific references and method-specific citations. |
| 📄 [License](https://zaedpolsl.github.io/HistoKit/license/) | Terms of use and third-party licensing information. |

## 📝 Citation

If you use HistoKit in your research, please cite our original publication:

> Merta, J., & Marczyk, M. (2026).  
> **HistoKit: Fast and Accurate Tissue and Artifact Detection and Data Processing for Whole Slide Histopathological Imaging.**  
> Proceedings of the 19th International Joint Conference on Biomedical
> Engineering Systems and Technologies (BIOSTEC 2026).

[![DOI](https://img.shields.io/badge/DOI-10.5220%2F0000222000004070-blue?style=flat-square)](https://doi.org/10.5220/0000222000004070)

<details>
<summary>📋 BibTeX citation</summary>

<br>

```bibtex
@inproceedings{Merta2026HistoKit,
  author    = {Merta, Julia and Marczyk, Micha{\l}},
  title     = {HistoKit: Fast and Accurate Tissue and Artifact Detection and Data Processing for Whole Slide Histopathological Imaging},
  booktitle = {Proceedings of the 19th International Joint Conference on Biomedical Engineering Systems and Technologies (BIOSTEC 2026), Volume 2: BIODEVICES, BIOIMAGING, BIOINFORMATICS},
  year      = {2026},
  month     = mar,
  address   = {Marbella, Spain},
  publisher = {SCITEPRESS -- Science and Technology Publications, Lda},
  isbn      = {978-989-758-802-0},
  issn      = {2184-4305},
  doi       = {10.5220/0000222000004070},
  url       = {https://doi.org/10.5220/0000222000004070}
}
```

</details>

For additional citations associated with individual methods, including
tissue segmentation post-processing, GrandQC, and stain normalization,
please refer to the
[full citation guide](https://zaedpolsl.github.io/HistoKit/citations/).

## 🙏 Acknowledgements

We gratefully acknowledge the authors of the following projects for
making their implementations and resources publicly available:

- [GrandQC](https://github.com/cpath-ukk/grandqc) — deep learning-based
  quality control and artifact segmentation in digital pathology.
- [StainTools](https://github.com/Peter554/StainTools) — stain normalization
  methods for histopathological images.

## 💬 Contact & Feedback

We welcome feedback, questions, and suggestions from the research community.

- 🐛 [Report an issue](https://github.com/ZAEDPolSl/HistoKit/issues)
- 💡 [Suggest a feature](https://github.com/ZAEDPolSl/HistoKit/issues/new)

If you find HistoKit useful in your research, consider giving this repository a ⭐.

## 📄 License

HistoKit is available for non-commercial use under the
[Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
(CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/) license.

For detailed terms of use, permitted applications, and third-party
licensing information, please refer to the
[License documentation](https://zaedpolsl.github.io/HistoKit/license/).

---

<p align="center">
  <strong>HistoKit</strong><br>
  <sub>Developed for digital and computational pathology research.</sub>
</p>

<p align="center">
  <a href="https://zaedpolsl.github.io/HistoKit/">Documentation</a> •
  <a href="https://github.com/ZAEDPolSl/HistoKit">GitHub</a> •
  <a href="https://doi.org/10.5220/0000222000004070">Publication</a>
</p>

