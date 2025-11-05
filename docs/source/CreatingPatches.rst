Creating Patches
===============================

This example shows how to remove gray stains from an H&E image.

.. code-block:: python

    from histo_kit.tissue_seg.utils import remove_gray_stains

    mask = remove_gray_stains(img)