HistoKit Documentation
======================

HistoKit is a Python package for processing Whole Slide Images (WSI).
It provides a comprehensive toolkit for common WSI operations:

Features
--------

- **Staining normalization** – standardize H&E images for consistent analysis.
- **Tissue detection** – identify tissue regions within slides.
- **Artifact detection** – detect artifacts using the **GrandQC** model.
- **Feature extraction** – extract image features using foundation models.
- **Patch extraction** – divide slides into patches at a specified zoom level and step size.
- **Visualization** – explore extracted features with **UMAP** embeddings.


Installation
------------

.. code-block:: bash

   pip install histo_kit

GitHub Repository
-----------------

You can find the source code on  🔗  `GitHub <https://github.com/ZAEDPolSl/HistoKit>`_

.. toctree::
   :maxdepth: 1
   :caption: Quick Start

   install

.. toctree::
   :maxdepth: 1
   :caption: Source Code

   git

Examples
--------

.. toctree::
   :maxdepth: 2
   :caption: Examples

   Examples/TissueDetection
   Examples/CreatingPatches

Documentation
-------------
.. toctree::
   :maxdepth: 1
   :caption: Reference

   api

