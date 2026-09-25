# HistoKit

HistoKit is a Python package for processing Whole Slide Images (WSI).  
It provides a toolkit for common operations in digital pathology workflows.

<img width="1461" height="821" alt="obraz" src="https://github.com/user-attachments/assets/5987b387-7b6d-4a5e-9a8d-bff75968955e" />


## Features

- **Tissue detection** — identifies tissue regions within whole-slide images.
- **Artifact detection** — uses the **GrandQC** deep learning model.
- **Patch extraction** — tiles slides into patches at customizable magnification and stride.
- **Augmentation & Stain Normalisation** — applies common augmentation methods and stain normalisation techniques (using staintools library).
  
## Installation
### From remote repository

The first way to use HistoKit is to download the source code from a remote repository. It is recommended to create a virtual environment with Anaconda or Venv. Then it is necessary to install the required dependencies from the requirements.txt file. Which can be done with the following command:

```bash
pip install -r requirements.txt
```

Then OpenSlide library needs to be installed with:

```bash
pip install openslide-python
```

If everything has been installed correctly, now it is possible to use functions from the HistoKit package after extracting files from a .zip archive.

### As a local pip package

It is also possible to install the package from .zip archive with pip. To do that repeat the steps described in the previous section, but now you don’t need to extract the files. You just have to install HistoKit from your .zip file with the following command:

```bash
pip install {path-to-histokit-archive}.zip
```

## Documentation and Examples

[HistoKit Documentation](https://zaedpolsl.github.io/HistoKit)

## References

> [!IMPORTANT]
> **If you use **HistoKit** in your research, according to the license, you must cite the following publications.**
> **HistoKit is provided for non-commercial, academic use only.**
> Any use of HistoKit components is subject to the terms of the original license. Please review it before using these components.
> The authors assume no responsibility for any issues with running the software, implementation errors, or other problems related to its operation.

#### HistoKit 1.0.0


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

#### Tissue Segmentation Post-processing

If you use the tissue segmentation post-processing methods included in HistoKit, please cite:

```bibtex
@inproceedings{Marczyk2025PostProcessing,
  author    = {Marczyk, M. and Wrobel, A. and Merta, J. and Polanska, J.},
  title     = {Post-Processing of Thresholding or Deep Learning Methods for Enhanced Tissue Segmentation of Whole-Slide Histopathological Images},
  booktitle = {Proceedings of the 18th International Joint Conference on Biomedical Engineering Systems and Technologies (BIOIMAGING 2025), Volume 1},
  pages     = {229--238},
  publisher = {SciTePress},
  year      = {2025},
  isbn      = {978-989-758-731-3},
  doi       = {10.5220/0013174700003911},
  url       = {https://doi.org/10.5220/0013174700003911},
  note      = {GitHub: https://github.com/ZAEDPolSl/WSI_TissueSeg}
}
```



### GrandQC-based Artifact Segmentation

If you use the **GrandQC-based artifact segmentation model**, GrandQC itself, or the associated benchmark in your research, you must cite the original GrandQC publication:

> Weng, Z. et al. *GrandQC: A comprehensive solution to quality control problem in digital pathology*.
> **Nature Communications** 15, 10685 (2024).
> DOI: https://doi.org/10.1038/s41467-024-54769-y

We would like to thank GrandQC Authors for making their implementation and model weights publicly available.

```bibtex
@article{Weng2024GrandQC,
  author  = {Weng, Z. and Seper, A. and Pryalukhin, A. and others},
  title   = {GrandQC: A comprehensive solution to quality control problem in digital pathology},
  journal = {Nature Communications},
  volume  = {15},
  pages   = {10685},
  year    = {2024},
  doi     = {10.1038/s41467-024-54769-y},
  url     = {https://doi.org/10.1038/s41467-024-54769-y},
  note    = {GitHub: https://github.com/cpath-ukk/grandqc}
}
```

For more information about GrandQC, see the original repository:

* **GrandQC repository:** https://github.com/cpath-ukk/grandqc
* **GrandQC license:** https://github.com/cpath-ukk/grandqc?tab=License-1-ov-file

> [!IMPORTANT]
> **GrandQC is provided for non-commercial, academic use only.**
> Any use of GrandQC-based components is subject to the terms of the original GrandQC license. Please review the GrandQC license before using these components.
