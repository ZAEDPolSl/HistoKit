# Citations

If you use **HistoKit** in your research, please cite the original HistoKit publication.

Depending on the functionalities used in your workflow, please also cite the relevant publications listed below, including those describing tissue segmentation post-processing, GrandQC-based artifact segmentation, and stain normalization methods.

## HistoKit (v1.0.0)

Please cite the following publication when using HistoKit:

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

## Tissue Segmentation Post-processing

If you use the tissue segmentation methods implemented in HistoKit, please also cite:

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

## GrandQC-based Artifact Segmentation

If you use the GrandQC-based artifact segmentation functionality included in HistoKit, please cite the original GrandQC publication:

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

We thank the GrandQC authors for making their implementation and pretrained model weights publicly available.

**Important:** GrandQC is provided for non-commercial academic use only and is subject to its own licensing terms.

- GrandQC repository: https://github.com/cpath-ukk/grandqc
- GrandQC license: https://github.com/cpath-ukk/grandqc?tab=License-1-ov-file

## Stain Normalization

If you use stain normalization methods available in HistoKit, please cite the corresponding original publications.

### Macenko Normalization

```bibtex
@inproceedings{Macenko2009,
  author    = {Macenko, Marc and Niethammer, Marc and Marron, J. S. and Borland, David and Woosley, John T. and Guan, Xiaojun and Schmitt, Charles and Thomas, Nancy E.},
  title     = {A method for normalizing histology slides for quantitative analysis},
  booktitle = {2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro},
  pages     = {1107--1110},
  year      = {2009},
  doi       = {10.1109/ISBI.2009.5193250}
}
```

### Vahadane Normalization

```bibtex
@article{Vahadane2016,
  author  = {Vahadane, Abhishek and Peng, Tingying and Sethi, Alok and Albarqouni, Shadi and Wang, Lichao and Baust, Maximilian and Steiger, Katja and Schlitter, Anna Melissa and Esposito, Irene and Navab, Nassir},
  title   = {Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images},
  journal = {IEEE Transactions on Medical Imaging},
  volume  = {35},
  number  = {8},
  pages   = {1962--1971},
  year    = {2016},
  doi     = {10.1109/TMI.2016.2529665}
}
```

### Reinhard Normalization

```bibtex
@article{Reinhard2001,
  author  = {Reinhard, Erik and Adhikhmin, Michael and Gooch, Bruce and Shirley, Peter},
  title   = {Color transfer between images},
  journal = {IEEE Computer Graphics and Applications},
  volume  = {21},
  number  = {5},
  pages   = {34--41},
  year    = {2001},
  doi     = {10.1109/38.946629}
}
```

### StainTools Acknowledgement

We acknowledge Peter Byfield for making StainTools publicly available and providing an open-source implementation of stain normalization methods for histopathology images.

StainTools repository: https://github.com/Peter554/StainTools

## Licensing and Attribution

HistoKit is distributed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)** license.

HistoKit may be used for non-commercial academic research, subject to the applicable license terms. When publishing research results obtained using HistoKit, please cite the original HistoKit publication and any additional publications relevant to the methods used.

Third-party components, including GrandQC, remain subject to their respective licenses.

For additional information, please refer to the [HistoKit license](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en).