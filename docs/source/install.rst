Installation
======================

From remote repository
----------------------
The first way to use HistoKit is to download the source code from a remote repository.
It is recommended to create a virtual environment with Anaconda or venv.
Then it is necessary to install the required dependencies from the requirements.txt file.
Which can be done with the following commands:

.. code-block:: bash

   pip install -r requirements.txt

Then OpenSlide library needs to be installed separately with:

.. code-block:: bash

   pip install openslide-bin

If everything has been installed correctly, now it is possible to use functions
from the HistoKit package after extracting files from a .zip archive.

As a local pip package
----------------------

It is also possible to install the package from .zip archive with pip. To do that repeat the steps described in the previous section, but now you don't need to extract the files.
You just have to install histo_kit from your .zip file with the following command:


.. code-block:: bash

   pip install {path-to-histokit-archive}.zip