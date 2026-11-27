### GaMRed-based Tissue Segmentation

HistoKit implements a tissue segmentation method based on the **GaMRed algorithm** (Marczyk et al., 2020), which combines Gaussian Mixture Model (GMM) decomposition of color intensity distributions with adaptive thresholding and morphological post-processing. The method was selected for its robustness to inter-slide color variability and effective refinement of tissue segmentation masks.

#### Segmentation algorithm

The tissue detection pipeline consists of the following steps:

1. **Histogram decomposition** – intensity histograms are generated independently for each RGB color channel and approximated using a two-component Gaussian Mixture Model. The two components are intended to represent tissue and background pixels.

2. **Thresholding** – a threshold separating tissue from background is determined independently for each color channel. If the estimated threshold falls below a predefined minimum (178.5 by default), a two-step Otsu thresholding procedure is used as a fallback.

3. **Tissue vs Background segmentation** – pixels with intensity values exceeding the corresponding thresholds in at least two RGB channels are classified as background.

4. **Artifact removal** – optional filtering enables the exclusion of low-chroma artifacts and black marker annotations.

5. **Morphological post-processing** – the resulting binary tissue mask is refined using morphological closing, opening, and hole filling.

6. **Small region removal** – small regions are removed using an adaptive area threshold. The initial threshold can be defined by the user or automatically estimated as the total mask area raised to the power of 0.45. The threshold is further refined using k-means clustering (k = 2) of logarithmically transformed connected-component areas. 
To identify two clusters: one with large tissue regions and the second one with smaller regions.
The minimum area of the cluster representing larger tissue regions, minus one pixel, determines the final removal threshold. 
This approach adapts to the distribution of tissue region sizes rather than relying on a fixed proportion of the slide area.

#### Configuration file

## Configuration

The GaMRed-based tissue detection pipeline is configured using the
`config/gamred.yaml` file. The configuration allows users to customize
processing magnification, tissue detection parameters, artifact removal,
morphological post-processing, and output generation.

### Configuration file

```yaml title="config/gamred.yaml" linenums="1"
# output
saver: hdf5
save_mag: 10.0
out_dir: ./outputs

# magnification settings
vis_mag: 2.5
tissdet_mag: 4

# tissue detection
thr_min: 178.5
split_regions: true

# green pen removal
remove_green_pen: true
thr_green_pen: [15, 120]
disk_radius_green_pen: 9

# black pen removal
remove_black_pen: true
thr_black_pen: [15, 0]
disk_radius_black_pen: 9

# gray stain removal
remove_gray_stains: true

# postprocessing pipeline
postprocess_steps:
  - name: Opening
    params:
      disk_radius: 2

  - name: FillHoles
    params:
      enabled: true

  - name: Opening
    params:
      disk_radius: 2

  - name: RemoveSmallRegions
    params:
      thr_area: null

# outputs / visualizations
collectors:
  - name: ThumbnailCollector

  - name: SegmentationOverlayCollector

  - name: HistogramCollector

  - name: MaskCollector
```

### Configuration parameters

#### :material-content-save: Output settings

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `saver` | `str` | `hdf5` | Output storage format. |
| `save_mag` | `float` | `10.0` | Magnification at which the output segmentation mask is saved. |
| `out_dir` | `str` | `./outputs` | Directory where processing results are stored. |

#### :material-magnify: Magnification settings

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `vis_mag` | `float` | `2.5` | Magnification used to generate visualization outputs. |
| `tissdet_mag` | `float` | `4` | Magnification at which tissue segmentation and mask post-processing are performed. |

!!! note "Processing magnification"

    Magnification affects the spatial resolution of the segmentation mask,
    computational cost, and the effective size of morphological structuring
    elements. Parameters expressed in pixels may require adjustment when
    changing the tissue detection magnification.

#### :material-microscope: Tissue detection

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `thr_min` | `float` | `178.5` | Minimum acceptable intensity threshold estimated using Gaussian Mixture Models. If the estimated threshold falls below this value, two-step Otsu thresholding is used as a fallback. |
| `split_regions` | `bool` | `true` | Enables tissue region splitting. |

#### :material-marker: Green pen removal

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `remove_green_pen` | `bool` | `true` | Enables green pen marking removal. |
| `thr_green_pen` | `list[int]` | `[15, 120]` | Threshold values used to identify green pen markings. |
| `disk_radius_green_pen` | `int` | `9` | Radius of the disk-shaped structuring element used during green pen removal, expressed in pixels. |

#### :material-marker: Black pen removal

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `remove_black_pen` | `bool` | `true` | Enables black pen marking removal. |
| `thr_black_pen` | `list[int]` | `[15, 0]` | Threshold values used to identify black pen markings. |
| `disk_radius_black_pen` | `int` | `9` | Radius of the disk-shaped structuring element used during black pen removal, expressed in pixels. |

#### :material-filter-outline: Gray stain removal

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `remove_gray_stains` | `bool` | `true` | Enables the removal of low-chroma gray artifacts from the tissue segmentation mask. |

### :material-tune: Post-processing pipeline

The `postprocess_steps` section defines the sequence of operations
applied to the initial binary tissue mask.

Each operation is specified by its `name` and an optional `params`
dictionary containing operation-specific parameters.

Operations are executed sequentially in the order defined in the
configuration file. The same operation may be included multiple times
with different parameter values.

#### Opening

Morphological opening removes small foreground structures and thin
connections between tissue regions. It consists of erosion followed
by dilation using a disk-shaped structuring element.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `disk_radius` | `int` | `2` | Radius of the disk-shaped structuring element, expressed in pixels. |

#### FillHoles

Fills enclosed background regions within the binary tissue mask.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `enabled` | `bool` | `true` | Enables hole filling. |

#### RemoveSmallRegions

Removes small connected components using an adaptive area-based
filtering strategy.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `thr_area` | `int`, `float` or `null` | `null` | Initial connected-component area threshold, expressed in pixels. If `null`, the threshold is estimated automatically from the mask dimensions. |

When `thr_area` is set to `null`, the initial threshold is calculated as:

$$
T_{\mathrm{initial}} =
\operatorname{round}\left((H \cdot W)^{0.45}\right)
$$

where $H$ and $W$ denote the height and width of the binary mask.

The initial threshold may be further refined using k-means clustering
($k=2$) of logarithmically transformed connected-component areas.

One cluster represents relatively large
tissue regions. The final threshold is determined as the minimum
component area within this cluster minus one pixel.

If no component exceeds the initial threshold, clustering is performed
using all connected-component areas.

Finally, connected components with areas less than or equal to the
resulting threshold are removed using 8-connectivity.

!!! info "Adaptive area threshold"

    The `thr_area` parameter specifies an initial threshold rather than
    a fixed minimum object area. Even when provided explicitly, its value
    may be adjusted by the adaptive clustering procedure.

### :material-chart-box-outline: Outputs and visualizations

The `collectors` section defines which processing results and
visualizations are collected during tissue detection.

| Collector | Description |
| :--- | :--- |
| `ThumbnailCollector` | Generates a thumbnail of the original whole-slide image. |
| `SegmentationOverlayCollector` | Generates a visualization of the tissue segmentation mask overlaid on the original slide. |
| `HistogramCollector` | Generates RGB intensity histograms with the corresponding tissue detection thresholds. |
| `MaskCollector` | Collects the resulting binary tissue segmentation mask (white - tissue, black - background).|

The list of collectors can be customized according to the required
processing outputs and quality control needs.

!!! tip "Quality control"

    Use `SegmentationOverlayCollector` and `HistogramCollector` to
    visually inspect tissue detection results and evaluate the selected
    segmentation thresholds.



## :material-format-quote-close: References and citations

The tissue detection method implemented in HistoKit is based on the
GaMRed algorithm, originally developed for adaptive filtering of
high-throughput biological data using Gaussian mixture decomposition,
and incorporates tissue segmentation post-processing methods.

If you use GaMRed-based tissue detection in your research, please cite
the following publications.

### GaMRed algorithm

> Marczyk, M., Jaksik, R., Polanski, A., & Polanska, J. (2020).
> **GaMRed-Adaptive Filtering of High-Throughput Biological Data.**
> IEEE/ACM Transactions on Computational Biology and Bioinformatics,
> 17(1), 149–157.

[:material-file-document-outline: Read the publication](https://doi.org/10.1109/TCBB.2018.2858825) |
[:material-database-search: PubMed](https://pubmed.ncbi.nlm.nih.gov/30040660/)

??? quote "BibTeX citation"

    ```bibtex
    @article{Marczyk2020GaMRed,
      author  = {Marczyk, Michal and Jaksik, Roman and Polanski, Andrzej and Polanska, Joanna},
      title   = {{GaMRed}-Adaptive Filtering of High-Throughput Biological Data},
      journal = {IEEE/ACM Transactions on Computational Biology and Bioinformatics},
      year    = {2020},
      volume  = {17},
      number  = {1},
      pages   = {149--157},
      doi     = {10.1109/TCBB.2018.2858825},
      url     = {https://doi.org/10.1109/TCBB.2018.2858825}
    }
    ```

### Tissue segmentation post-processing

> Marczyk, M., Wrobel, A., Merta, J., & Polanska, J. (2025).
> **Post-Processing of Thresholding or Deep Learning Methods for
> Enhanced Tissue Segmentation of Whole-Slide Histopathological Images.**
> Proceedings of the 18th International Joint Conference on Biomedical
> Engineering Systems and Technologies (BIOIMAGING 2025),
> Volume 1, pp. 229–238.

[:material-file-document-outline: Read the publication](https://doi.org/10.5220/0013174700003911) |
[:fontawesome-brands-github: Original repository](https://github.com/ZAEDPolSl/WSI_TissueSeg)

??? quote "BibTeX citation"

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

For the main HistoKit publication and additional method-specific
references, please refer to the
[:material-book-open-variant: HistoKit citation guide](https://zaedpolsl.github.io/HistoKit/citations/).