# Cohort processing

HistoKit provides a configurable framework for cohort-level processing
of whole-slide images (WSIs), enabling automated and reproducible
analysis of large histopathological datasets.

Instead of executing individual processing methods separately, users
can define a multi-stage processing pipeline using a YAML files.

The main file for WSI processing `configs/cohort_wsi.yaml` contains
parameters of cohort processing. Parameters for each step can be 
modified in the configuration file of each algorithm.

## Processing workflow

The cohort processing pipeline consists of independent stages that
can be enabled or disabled depending on the analysis requirements.

| Stage | Algorithm | Description |
| :--- | :--- | :--- |
| :material-microscope: Tissue detection | GaMRed | Identifies tissue regions within whole-slide images. |
| :material-brain: Artifact detection | GrandQC | Detects histological artifacts using deep learning-based segmentation. |
| :material-chart-box-outline: Statistics | Stats | Computes quantitative statistics based on segmentation results. |
| :material-vector-polygon: GeoJSON | GeoJSON | Exports segmentation regions as vector-based annotations in GeoJSON format. |

Each stage has its own configuration file, allowing processing
parameters to be adjusted independently.

Previously generated segmentation masks can also be supplied as
input to subsequent processing stages, without repeating the
entire processing pipeline.

## Configuration

Cohort processing is controlled through the `cohort_wsi.yaml`
configuration file.

The configuration defines:

- Output storage format.
- Logging settings.
- Enabled processing stages.
- Algorithm selection.
- Paths to stage-specific configuration files.
- Number of parallel workers.
- Overwrite behavior.
- Locations of previously generated segmentation masks.

### Configuration file

```yaml title="cohort_wsi.yaml" linenums="1"
saver: hdf5

logging:
  enabled: true
  log_dir: ./outputs/logs
  processed_file: processed.csv
  errors_file: errors.csv

stages:
  tissue_detection:
    enabled: false
    algorithm: gamred
    config_path: ./gamred.yaml
    parallel_workers: 4
    overwrite: false

  artifact_detection:
    enabled: false
    algorithm: grandqc
    config_path: ./grandqc.yaml
    parallel_workers: 1
    overwrite: false
    tissue_source:
      dir: null

  statistics:
    algorithm: stats
    config_path: ./stats.yaml
    enabled: false
    parallel_workers: 1
    artifact_source:
      dir: /path/to/artifact_detection/grandqc/masks/

  geojson:
    algorithm: geojson
    config_path: ./geojson.yaml
    enabled: true
    parallel_workers: 3
    artifact_source:
      dir: /path/to/artifact_detection/grandqc/masks/
```

!!! note "Example configuration"

    In the configuration above, only the GeoJSON export stage is enabled.
    Previously generated artifact segmentation masks are used as input.

    Enable additional stages by setting their `enabled` parameter
    to `true`.

## Configuration parameters

### :material-content-save: General settings

| Parameter | Type | Example value | Description |
| :--- | :--- | :--- | :--- |
| `saver` | `str` | `hdf5` | Storage format used for processing results. |
| `stages` | `dict` | — | Defines the individual processing stages and their configuration. |

### :material-file-document-outline: Logging

Logging provides information about cohort processing progress
and helps identify slides that could not be processed successfully.

| Parameter | Type | Example value | Description |
| :--- | :--- | :--- | :--- |
| `enabled` | `bool` | `true` | Enables processing logs. |
| `log_dir` | `str` | `./outputs/logs` | Directory where log files are stored. |
| `processed_file` | `str` | `processed.csv` | CSV file containing information about successfully processed slides. |
| `errors_file` | `str` | `errors.csv` | CSV file containing information about processing errors. |

### :material-tune: Common stage parameters

Each processing stage can be configured independently using
the following parameters.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `enabled` | `bool` | Determines whether the processing stage is executed. |
| `algorithm` | `str` | Identifier of the processing algorithm. |
| `config_path` | `str` | Path to the algorithm-specific YAML configuration file. |
| `parallel_workers` | `int` | Number of parallel workers assigned to the processing stage. |
| `overwrite` | `bool` | Determines whether existing processing results should be overwritten, where supported. |

!!! tip "Parallel processing"

    The number of parallel workers can be configured separately
    for each processing stage.

    CPU-based operations may benefit from multiple workers,
    whereas GPU-based artifact segmentation may require fewer
    workers due to GPU memory limitations.

    Adjust these settings according to the available hardware,
    slide dimensions, and memory requirements.

## Processing stages

### :material-microscope: Tissue detection

The tissue detection stage identifies tissue regions and separates
them from the slide background.

The GaMRed-based pipeline supports adaptive thresholding,
artifact removal, morphological post-processing, and adaptive
filtering of small connected components.

```yaml
tissue_detection:
  enabled: true
  algorithm: gamred
  config_path: ./gamred.yaml
  parallel_workers: 4
  overwrite: false
```

Detailed algorithm parameters are defined in `gamred.yaml`.

### :material-brain: Artifact detection

The artifact detection stage uses GrandQC-based segmentation
to identify histological artifacts within whole-slide images.

```yaml
artifact_detection:
  enabled: true
  algorithm: grandqc
  config_path: ./grandqc.yaml
  parallel_workers: 1
  overwrite: false
  tissue_source:
    dir: /path/to/tissue/masks/
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `tissue_source.dir` | `str` or `null` | Directory containing previously generated tissue segmentation masks, if provided. |

Detailed model settings and segmentation parameters are defined
in `grandqc.yaml`.

!!! note "External model"

    GrandQC requires pretrained model weights and is subject to
    separate licensing and citation requirements.

    See the [Installation Guide](../../installation.md)
    for model download and configuration instructions.

### :material-chart-box-outline: Statistics

The statistics stage computes quantitative measurements based
on previously generated segmentation results.

```yaml
statistics:
  enabled: true
  algorithm: stats
  config_path: ./stats.yaml
  parallel_workers: 4
  artifact_source:
    dir: /path/to/artifact/masks/
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `artifact_source.dir` | `str` | Directory containing artifact segmentation masks used for statistical analysis. |

The calculated measurements and output settings are defined
in `stats.yaml`.

### :material-vector-polygon: GeoJSON export

The GeoJSON stage converts segmentation regions into vector-based
annotations that can be used in downstream digital pathology
workflows and compatible visualization tools.

```yaml
geojson:
  enabled: true
  algorithm: geojson
  config_path: ./geojson.yaml
  parallel_workers: 3
  artifact_source:
    dir: /path/to/artifact/masks/
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `artifact_source.dir` | `str` | Directory containing segmentation masks to be converted into GeoJSON annotations. |

Export parameters are defined in `geojson.yaml`.

## Example workflows

### Full cohort processing

Enable all processing stages to perform tissue detection,
artifact detection, statistical analysis, and GeoJSON export.

```yaml
stages:
  tissue_detection:
    enabled: true

  artifact_detection:
    enabled: true

  statistics:
    enabled: true

  geojson:
    enabled: true
```

### Tissue detection only

Use this configuration to generate tissue segmentation masks
without performing artifact detection or subsequent analysis.

```yaml
stages:
  tissue_detection:
    enabled: true

  artifact_detection:
    enabled: false

  statistics:
    enabled: false

  geojson:
    enabled: false
```

### Analysis of existing segmentation masks

If artifact segmentation masks have already been generated,
the statistics and GeoJSON stages can be executed independently.

```yaml
stages:
  tissue_detection:
    enabled: false

  artifact_detection:
    enabled: false

  statistics:
    enabled: true
    artifact_source:
      dir: /path/to/artifact/masks/

  geojson:
    enabled: true
    artifact_source:
      dir: /path/to/artifact/masks/
```

!!! info "Configuration examples"

    The examples above show only the parameters that need to
    be adjusted in the complete `cohort_wsi.yaml` configuration.
    They are not standalone configuration files.

## Running cohort processing

Before starting cohort processing:

1. Prepare the input whole-slide image dataset.
2. Configure the `cohort_wsi.yaml` file.
3. Enable the required processing stages.
4. Adjust the algorithm-specific configuration files.
5. Specify input directories for previously generated masks,
   if required.
6. Adjust the number of parallel workers.
7. Run the cohort processing pipeline.

### Python example

<!-- TODO: Add the cohort processing entry point and usage example. -->

## Outputs

The generated outputs depend on the enabled processing stages,
the selected storage format, and the configuration of individual
algorithms.

| Output | Description |
| :--- | :--- |
| :material-image-outline: Tissue masks | Binary masks identifying tissue regions. |
| :material-image-filter-center-focus: Artifact masks | Segmentation masks identifying histological artifacts. |
| :material-image-multiple-outline: Visualizations | Thumbnails, segmentation overlays, and other quality control outputs, depending on the configured collectors. |
| :material-chart-box-outline: Statistics | Quantitative measurements derived from segmentation results. |
| :material-vector-polygon: GeoJSON annotations | Vector representations of segmentation regions. |
| :material-file-check-outline: Processing logs | Information about successfully processed slides. |
| :material-file-alert-outline: Error logs | Information about processing failures. |

### Output directory structure

An illustrative output directory structure is presented below.
The actual directory structure depends on the stage-specific
configuration files and enabled processing stages.

```text
outputs/
├── tissue_detection/
│   └── gamred/
│       ├── masks/
│       └── visualizations/
│
├── artifact_detection/
│   └── grandqc/
│       ├── masks/
│       └── visualizations/
│
├── statistics/
│
├── geojson/
│
└── logs/
    ├── processed.csv
    └── errors.csv
```

## Recommended practices

!!! tip "Processing large cohorts"

    Before processing an entire cohort:

    - Test the configuration on a representative subset of slides.
    - Inspect tissue and artifact segmentation visualizations.
    - Verify the generated outputs.
    - Adjust processing magnification and algorithm parameters.
    - Select the number of parallel workers according to available
      computational resources.
    - Enable logging to find potential errors during processing.

    Once the configuration has been validated, the same settings
    can be applied consistently across the entire cohort.