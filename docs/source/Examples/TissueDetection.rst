Tissue & Artifacts Detection
===============================


Run with CUDA (and CPU with a single thread)
--------------------------------------------

To run the program with CUDA, use the ``src/run_tissue_seg_single.py`` script and set the ``device`` parameter to ``cuda``.

Run detection on many threads (CPU)
-----------------------------------

To run the program with multiple threads, use the ``src/run_tissue_seg.py`` script and set the ``workers`` parameter to the number of threads you want to use. Using ``cuda`` is not recommended for multiple workers due to competition for resources.

Configuration
-------------

Parameters
^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 15 10 25 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``--wsi_dir``
     - str
     - ``../data/``
     - Input directory with WSIs (whole-slide images saved as SVS files).
   * - ``--out_dir``
     - str
     - ``../res/``
     - Output directory for results.
   * - ``--split_regions``
     - bool
     - True
     - If multiple regions are present on the slide, save each region separately.
   * - ``--fill_holes``
     - bool
     - False
     - Fill holes in the tissue mask or not.
   * - ``--close_disk_r``
     - int
     - 2
     - Radius of disk structuring element used for closing operation (mask cleaning).
   * - ``--open_disk_r``
     - int
     - 2
     - Radius of disk structuring element used for opening operation (mask cleaning).
   * - ``--save_mask_formats``
     - list
     - [``npy``, ``mat``]
     - File formats for saving masks. Choose at least one: ``npy``, ``mat``.
   * - ``--device``
     - str
     - cpu
     - Device for artifacts detection: ``cuda`` or ``cpu`` (only for ``run_tissue_seg_single.py``)
   * - ``--grandqc_model``
     - str
     - ``grand_qc/models/GrandQC_MPP1.pth``
     - Path to GrandQC model weights (10x magnification model used by default).
   * - ``--workers``
     - int
     - 10
     - Number of workers for parallel processing (max number: ``os.cpu_count()``).

Output Folders
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Folder name
     - Description
   * - masks/
     - Masks with detected tissue and GrandQC results (saved as `.npy`, `.mat`, or both).
   * - bg_masks_vis/
     - Detected tissue region masks as small PNG thumbnails.
   * - bg_thr_hist/
     - Histograms of background thresholds used for tissue detection (PNG).
   * - raw_small/
     - Small PNG thumbnails of tissue images.
   * - pen_vis/
     - Results of pen removal (small PNG thumbnails).
   * - bg_removal_vis/
     - Background removal results (small PNG thumbnails).
   * - bg_removal_contour_vis/
     - Background removal with blue contours (small PNG thumbnails).
   * - grandqc_map_vis/
     - Artifact detection results from GrandQC as color maps (small PNG thumbnails).
   * - grandqc_overlay_vis/
     - Artifact detection results overlaid on tissue regions (small PNG thumbnails).
   * - grandqc_vis_region/
     - Artifact detection results for each tissue region (color maps in PNG).

Artifacts Color Mapping
-----------------------

.. image:: https://github.com/user-attachments/assets/3e8726c1-3b24-4505-92dc-04ed852d28e7
   :width: 452
   :height: 190

Example Results
---------------

masks
^^^^^^

.npz or .mat files are saved here.

bg_masks_vis
^^^^^^^^^^^^^

Small mask visualisations, tissue region is white (255), background is black (0).

.. image:: https://github.com/user-attachments/assets/0d0ab825-59b5-46ce-a409-da30b36fbaf2
   :width: 445
   :height: 276
   :align: center

bg_thr_hist
^^^^^^^^^^^^

Histograms with background threshold values for each color channel calculated with GaMRed algorithm, or with Otsu method when threshold obtained with GaMRed is too small.

.. image:: ../bg_thr_hist.png
   :width: 300
   :height: 400
   :align: center

raw_small
^^^^^^^^^^

Small tissue thumbnails.

.. image:: ../raw_small.png
   :width: 445
   :height: 276
   :align: center

pen_vis
^^^^^^^^

Small thumbnail with detected black pen regions.

.. image:: ../pen_vis.png
   :width: 445
   :height: 445
   :align: center

bg_removal_vis
^^^^^^^^^^^^^^^

Small thumbnail of the tissue region with removed background.

.. image:: ../bg_removal_vis.png
   :width: 445
   :height: 276
   :align: center

bg_removal_contour_vis
^^^^^^^^^^^^^^^^^^^^^^^

Contours visualisation of the detected tissue regions. Contours are marked in blue.

.. image:: ../bg_removal_contour_vis.png
   :width: 445
   :height: 276
   :align: center

grandqc_map_vis
^^^^^^^^^^^^^^^^

Visualisation of results obtained with GrandQC based on the tissue region detection map from GaMRed or Otsu algorithms.

.. image:: ../grandqc_map_vis.png
   :width: 445
   :height: 276
   :align: center

grandqc_overlay_vis
^^^^^^^^^^^^^^^^^^^^

Visualisation of the tissue region with contours of tissue area detected by GaMRed or Otsu algorithms marked in blue, and GrandQC results overlay.

.. image:: ../grandqc_overlay_vis.png
   :width: 445
   :height: 276
   :align: center

grandqc_vis_region
^^^^^^^^^^^^^^^^^^^

Images with tissue regions are saved, background is black while other areas are colored by GrandQC. Each region is saved separately. Bounding boxes are stored in `.mat` or `.npz` files.

.. image:: https://github.com/user-attachments/assets/63248351-083c-4725-8cf1-a1b46aad8b29
   :width: 1000
   :height: 800
   :align: center

Output `.mat` and `.npz` Files Descriptions
-------------------------------------------

For a whole image:

.. list-table::
   :header-rows: 1
   :widths: 15 50

   * - Key
     - Description
   * - basename
     - Tissue file basename (without the `.svs` extension)
   * - mask_all
     - Mask of detected tissue regions - 2D array with ``True`` for tissue and ``False`` for background
   * - mask_art
     - Mask of artifacts detected by GrandQC for the given region - 2D array with values 0–7
   * - ind_WSI
     - Indexes for WSI image layers (MATLAB-style indexing from 1)
   * - ratio
     - Ratio for each image layer
   * - scale_val
     - Scale factor applied to masks
   * - thr
     - Thresholds for R, G, B channels

For splitted regions:

.. list-table::
   :header-rows: 1
   :widths: 15 50

   * - Key
     - Description
   * - basename
     - Tissue file basename (without the `.svs` extension)
   * - mask_all
     - Cell array of 2D arrays for each region
   * - mask_art
     - Cell array of artifact masks for each region
   * - ind_WSI
     - Indexes for WSI image layers
   * - ratio
     - Ratio for each image layer
   * - scale_val
     - Scale factor applied to masks
   * - thr
     - Thresholds for R, G, B channels
   * - tiss_stats
     - Bounding box coordinates for each region

How to Load Regions to Matlab
-----------------------------

.. code-block:: matlab

    function [img,mask_all] = load_tiss_masked_python(svs_name,mask_name,reg_ID,qual_ind)
        load([mask_name],'mask_all','ratio','tiss_stats','ind_WSI','scale_val')
        scale_val = scale_val/ratio(ind_WSI == qual_ind);
        Box = tiss_stats(reg_ID,:)* scale_val;
        region = {[Box(1),Box(3)], [Box(2),Box(4)]};
        img = imread(svs_name,'Index',qual_ind,'PixelRegion',region);
        mask_all = uint8(imresize(mask_all{reg_ID},[size(img,1),size(img,2)]));
        for b=1:size(img,3)
            tmp = img(:,:,b).*mask_all;
            tmp(mask_all == 0) = 255;
            img(:,:,b) = tmp;
        end
        clear tmp
        figure;image(img)
    end

References
----------

- Weng, Z., Seper, A., Pryalukhin, A. et al.
  *GrandQC: A comprehensive solution to quality control problem in digital pathology.*
  Nature Communications 15, 10685 (2024).
  `DOI <https://doi.org/10.1038/s41467-024-54769-y>`_ | `GitHub <https://github.com/cpath-ukk/grandqc>`_

- Marczyk, M., Wrobel, A., Merta, J. and Polanska, J. (2025).
  *Post-Processing of Thresholding or Deep Learning Methods for Enhanced Tissue Segmentation of Whole-Slide Histopathological Images.*
  In Proceedings of the 18th International Joint Conference on Biomedical Engineering Systems and Technologies - Volume 1: BIOIMAGING;
  ISBN 978-989-758-731-3, SciTePress, pp. 229–238.
  `DOI <https://doi.org/10.5220/0013174700003911>`_ | `GitHub <https://github.com/ZAEDPolSl/WSI_TissueSeg>`_
