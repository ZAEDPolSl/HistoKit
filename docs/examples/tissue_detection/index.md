# Tissue detection

Tissue detection is a fundamental step in whole-slide image (WSI)
processing. Its primary objective is to distinguish tissue regions
from the slide background, enabling subsequent image analysis to
focus on relevant areas.

By excluding background regions, tissue detection reduces computational
cost, improves patch extraction efficiency, and supports downstream
computational pathology workflows.

HistoKit provides configurable tissue detection pipelines supporting:

- :material-microscope: Tissue and background separation.
- :material-magnify: Processing at user-defined magnification levels.
- :material-marker: Removal of pen markings and low-chroma artifacts.
- :material-tune: Customizable morphological post-processing.
- :material-chart-bell-curve: Adaptive removal of small tissue regions.
- :material-image-outline: Segmentation masks and quality control visualizations.
- :material-folder-multiple-outline: Batch processing of multiple whole-slide images.

## Available methods

| Method | Best for | Description |
| :--- | :--- | :--- |
| [GaMRed](gamred.md) | Tissue segmentation | Gaussian Mixture Model-based segmentation with adaptive thresholding, artifact removal, and configurable morphological post-processing. Designed to account for color variability between slides. |


## Choosing a method

The choice of tissue detection method and its configuration depends on
the characteristics of the input slides, including staining variability,
tissue morphology, image resolution, and the presence of artifacts.

