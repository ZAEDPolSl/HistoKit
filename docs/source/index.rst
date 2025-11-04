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

GitHub Repository
-----------------

You can find the source code on  🔗  `GitHub <https://github.com/ZAEDPolSl/HistoKit>`_


Installation
------------
From remote repository
~~~~~~~~~~~~~~~~~~~~~~
The first way to use HistoKit is to download the source code from a remote repository.
It is recommended to create a virtual environment with Anaconda or Venv.
Then it is necessary to install the required dependencies from the requirements.txt file.
Which can be done with the following command:

.. code-block:: bash

   pip install -r requirements.txt
   
Then OpenSlide library needs to be installed with:

.. code-block:: bash

   pip install openslide-bin

If everything has been installed correctly, now it is possible to use functions
from the HistoKit package after extracting files from a .zip archive.

As a local pip package
~~~~~~~~~~~~~~~~~~~~~~

It is also possible to install the package from .zip archive with pip. To do that repeat the steps described in the previous section, but now you don't need to extract the files.
You just have to install histo_kit from your .zip file with the following command:


.. code-block:: bash

   pip install {path-to-histokit-archive}.zip

Examples
--------

.. toctree::
   :maxdepth: 1
   :caption: Examples

   Examples/TissueDetection
   Examples/CreatingPatches

Documentation
-------------
.. toctree::
   :maxdepth: 1
   :caption: Reference

   api

