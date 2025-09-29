# Tools for processing WSI

## ⚙️ Installation
- Uses Python 3.10
```python
pip install -r requirements.txt
pip install openslide-bin
```
## ⚙️ Run with CUDA (single thread processing)

To run program with CUDA use `src/run_tissue_seg_single.py` script and set the `device` parameter to `cuda`.

## ⚙️ Run detection on many threads (CPU)

To run program with multiple threads use `src/run_tissue_seg.py` script and set the `workers` parameter to the number of threads you want to use.

## ⚙️ Configuration

### Parameters

| Parameter            | Type   | Default                                                                 | Description                                                                 |
|----------------------|--------|-------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| `--wsi_dir`          | str    | `../data/`                    | Input directory with WSIs (whole-slide images saved as SVS files).                             |
| `--out_dir`          | str    | `../res/`                                                    | Output directory for results.                                               |
| `--split_regions`    | bool   | `True`                                                                  | If multiple regions are present on the slide, save each region separately.  |
| `--fill_holes`       | bool   | `False`                                                                 | Fill holes in the tissue mask or not.                                       |
| `--close_disk_r`     | int    | `2`                                                                     | Radius of disk structuring element used for closing operation (mask cleaning). |
| `--open_disk_r`      | int    | `2`                                                                     | Radius of disk structuring element used for opening operation (mask cleaning). |
| `--save_mask_formats`| list   | `["npy", "mat"]`                                                        | File formats for saving masks. Choose at least one: `npy`, `mat`.           |
| `--device`           | str    | `cpu`                                                                   | Device for artifacts detection: `cuda` or `cpu`.                            |
| `--grandqc_model`    | str    | `grand_qc/models/GrandQC_MPP1.pth`                                      | Path to GrandQC model weights (10x magnification model used by default).    |
| `--workers`          | int    | `10`                                                                    | Number of workers for parallel processing (max number: `os.cpu_count()`).             |

### Output Folders

| Folder name               | Description                                                                        |
| ------------------------- | ---------------------------------------------------------------------------------- |
| `masks/`                  | Masks with detected tissue and GrandQC results (saved as `.npy`, `.mat`, or both). |
| `bg_masks_vis/`           | Detected tissue region masks as small PNG thumbnails.                              |
| `bg_thr_hist/`            | Histograms of background thresholds used for tissue detection (PNG).               |
| `raw_small/`              | Small PNG thumbnails of tissue images.                                             |
| `pen_vis/`                | Results of pen removal (small PNG thumbnails).                                     |
| `bg_removal_vis/`         | Background removal results (small PNG thumbnails).                                 |
| `bg_removal_contour_vis/` | Background removal with blue contours (small PNG thumbnails).                      |
| `grandqc_map_vis/`        | Artifact detection results from GrandQC as color maps (small PNG thumbnails).      |
| `grandqc_overlay_vis/`    | Artifact detection results overlaid on tissue regions (small PNG thumbnails).      |
| `grandqc_vis_region/`     | Artifact detection results for each tissue region (color maps in PNG).             |

### Artifacts color mapping

<img width="452" height="190" alt="Bez nazwy" src="https://github.com/user-attachments/assets/3e8726c1-3b24-4505-92dc-04ed852d28e7" />

### Output `.mat` and `.npz` files descriptions
#### For a whole image:
| Key          | Description |
|--------------|-------------|
| `basename`   | Tissue file basename (without the `.svs` extension). |
| `mask_all`   | Mask of detected tissue regions - 2d array with the whole image (filled with `True` (white) for tissue regions and `False` (black) for background) | 
| `mask_art`   | Mask of artifacts detected by GrandQC for the given region - 2d array with the whole image (filled with numbers from `0` to `7` corresponding to the detected artifact)|
| `ind_WSI`    | Indexes for WSI image layers (MATLAB-style indexing from 1). |
| `ratio`      | Ratio for each image layer. |
| `scale_val`  | Scale factor applied to masks. |
| `thr`        | Thresholds calculated for R, G, B color channels. |

#### For splitted regions:
| Key          | Description |
|--------------|-------------|
| `basename`   | Tissue file basename (without the `.svs` extension). |
| `mask_all`   | Mask of detected tissue regions - cell array of 2d arrays (filled with `True` (white) for tissue regions and `False` (black) for background) |
| `mask_art`   | Mask of artifacts detected by GrandQC for the given region - cell array of 2d arrays (filled with numbers from `0` to `7` corresponding to the detected artifact)|
| `ind_WSI`    | Indexes for WSI image layers (MATLAB-style indexing from 1). |
| `ratio`      | Ratio for each image layer. |
| `scale_val`  | Scale factor applied to masks. |
| `thr`        | Thresholds calculated for R, G, B color channels. |
| `tiss_stats` | Bounding box coordinates (in .mat files converted to MATLAB notation - indexing from 1). |

## How to load regions to Matlab?

```matlab
function [img,mask_all] = load_tiss_masked_python(svs_name,mask_name,reg_ID,qual_ind)
 
% load info about mask
load([mask_name],'mask_all','ratio','tiss_stats',...
    'ind_WSI','scale_val')
 
% calculate scaling value for selected image resolution
scale_val = scale_val/ratio(ind_WSI == qual_ind);
 
% get bounding box for the region and resize
%Box = tiss_stats.BoundingBox(reg_ID,:) * scale_val;
Box = tiss_stats(reg_ID,:)* scale_val;
 
% get location of the region and load image
region = {[Box(1),Box(3)], [Box(2),Box(4)]};
img = imread(svs_name,'Index',qual_ind,'PixelRegion',region);
% figure;image(img)
 
% get mask of region, resize and apply to image
mask_all = uint8(imresize(mask_all{reg_ID},[size(img,1),size(img,2)]));
for b=1:size(img,3)
    tmp = img(:,:,b).*mask_all;
    tmp(mask_all == 0) = 255;
    img(:,:,b) = tmp;
end
clear tmp
figure;image(img)
 
end
```

## 📚 References

- Weng, Z., Seper, A., Pryalukhin, A. et al.  
  *GrandQC: A comprehensive solution to quality control problem in digital pathology.*  
  Nature Communications 15, 10685 (2024).  
  [https://doi.org/10.1038/s41467-024-54769-y](https://doi.org/10.1038/s41467-024-54769-y)
  [Github repository](https://github.com/cpath-ukk/grandqc)

- Marczyk, M., Wrobel, A., Merta, J. and Polanska, J. (2025).  
  *Post-Processing of Thresholding or Deep Learning Methods for Enhanced Tissue Segmentation of Whole-Slide Histopathological Images.*  
  In Proceedings of the 18th International Joint Conference on Biomedical Engineering Systems and Technologies - Volume 1: BIOIMAGING;  
  ISBN 978-989-758-731-3, SciTePress, pp. 229–238.  
  [https://doi.org/10.5220/0013174700003911](https://doi.org/10.5220/0013174700003911)
  [Github repository](https://github.com/ZAEDPolSl/WSI_TissueSeg)

