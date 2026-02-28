import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.io as sio
from src.histo_kit.grand_qc.artifacts import Artifact


def calculate_stats(mask_mat_path):
    """
        Calculate statistics for each region in the given mask matrix.

        Args:
            mask_mat (dict): A dictionary containing the following keys:
                - 'mask_art': A 3D numpy array where the first dimension represents regions,
                              and the other two dimensions represent the artifact mask.
                - 'bbox': A list of bounding boxes for each region, where each bounding box
                          is a tuple (y0, x0, y1, x1).
                - 'basename': A string representing the base name of the image.
                - 'scale_val': A scaling value for the mask.
                - 'mask_mag': The magnification level of the mask.
                - 'mpp': Microns per pixel for the mask.
                - 'mag_l0': The level 0 magnification of the image.

        Returns:
            pd.DataFrame: A DataFrame containing statistics for each region, with the following columns:
                - 'basename': The base name of the image.
                - 'region_idx': The index of the region.
                - 'scale_val': The scaling value for the mask.
                - 'y0', 'x0', 'y1', 'x1': The bounding box coordinates for the region.
                - 'mask_mag': The magnification of the mask.
                - 'mpp': Microns per pixel for the mask.
                - 'mag_l0': The level 0 magnification of the wsi.
                - 'Total_pixels': The total number of pixels in the region.
                - 'Num_<Artifact>': The number of pixels for each artifact type.
                - 'Percent_<Artifact>': The percentage of pixels for each artifact type.

        Notes:
            - The function skips the `BG_THR` artifact when calculating statistics.
            - The artifact types are derived from the `Artifact` enumeration.

    """
    mask_mat = sio.loadmat(mask_mat_path)
    mask = mask_mat['mask_art']
    stats_list = []
    for idx, region in enumerate(mask[0]):
        region_mask = region > 0
        total = region_mask.sum()
        num_dict = {}
        percent_dict = {}

        for a in Artifact:
            if a == Artifact.BG_THR:
                continue
            count = np.sum(region == a.value)
            percentage = (count / total) * 100 if total > 0 else 0
            num_dict[a.name] = count
            percent_dict[a.name] = percentage

        bbox = mask_mat['bbox'][idx]
        stats_region = {'basename': mask_mat['basename'][0],
                        'region_idx': idx,
                        'scale_val': mask_mat['scale_val'][0][0],
                        'y0': bbox[0],
                        'x0': bbox[1],
                        'y1': bbox[2],
                        'x1': bbox[3],
                        'mask_mag': mask_mat['mask_mag'][0][0],
                        "mpp": mask_mat['mpp'][0][0],
                        "mag_l0": mask_mat['mag_l0'][0][0],
                        "Total_pixels": total,
                        "Num_NORM": num_dict["NORM"],
                        "Num_ART_FOLD": num_dict["ART_FOLD"],
                        "Num_ART_DARKSPOT": num_dict["ART_DARKSPOT"],
                        "Num_ART_PEN": num_dict["ART_PEN"],
                        "Num_ART_EDGE": num_dict["ART_EDGE"],
                        "Num_ART_FOCUS": num_dict["ART_FOCUS"],
                        "Num_BG_MODEL": num_dict["BG_MODEL"],
                        "Percent_NORM": percent_dict["NORM"],
                        "Percent_ART_FOLD": percent_dict["ART_FOLD"],
                        "Percent_ART_DARKSPOT": percent_dict["ART_DARKSPOT"],
                        "Percent_ART_PEN": percent_dict["ART_PEN"],
                        "Percent_ART_EDGE": percent_dict["ART_EDGE"],
                        "Percent_ART_FOCUS": percent_dict["ART_FOCUS"],
                        "Percent_BG_MODEL": percent_dict["BG_MODEL"],
                        "Percent_ART_ALL": 100 - (percent_dict["NORM"])}

        stats_list.append(stats_region)

    stats_region_df = pd.DataFrame(stats_list)
    stats_df = stats_region_df[['basename', 'scale_val', 'mask_mag', "mpp", "mag_l0", "Total_pixels", "Num_NORM", "Num_ART_FOLD", "Num_ART_DARKSPOT", "Num_ART_PEN", "Num_ART_EDGE", "Num_ART_FOCUS", "Num_BG_MODEL"]]
    stats_df = (stats_df.groupby(['basename'])[['Total_pixels', "Num_NORM", "Num_ART_FOLD",
        "Num_ART_DARKSPOT", "Num_ART_PEN", "Num_ART_EDGE", "Num_ART_FOCUS", "Num_BG_MODEL"]].sum().reset_index())

    art_names = ["NORM", "ART_FOLD", "ART_DARKSPOT", "ART_PEN", "ART_EDGE", "ART_FOCUS", "BG_MODEL"]

    stats_df[[f"Percent_{art}" for art in art_names]] = stats_df[[f"Num_{art}" for art in art_names]].div(stats_df["Total_pixels"], axis=0) * 100

    return stats_region_df, stats_df


def plot_stats(stats_df, output_path, title):
    art_names = ["NORM", "ART_FOLD", "ART_DARKSPOT",
                 "ART_PEN", "ART_EDGE", "ART_FOCUS", "BG_MODEL"]


    colors = [
        [128/255, 128/255, 128/255],  # NORM: gray
        [255/255, 99/255, 71/255],    # ART_FOLD: orange
        [0, 1, 0],                     # ART_DARKSPOT: green
        [1, 0, 0],                     # ART_PEN: red
        [1, 0, 1],                     # ART_EDGE: pink/magenta
        [75/255, 0, 130/255],          # ART_FOCUS: violet
        [50/255, 120/255, 230/255]     # BG_MODEL: blue
    ]

    pct_cols = [f"Percent_{art}" for art in art_names]

    data = [stats_df[col].dropna() for col in pct_cols]

    fig, ax = plt.subplots(figsize=(12, 8))

    bp = ax.boxplot(data, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    ax.set_xticks(range(1, len(art_names) + 1))
    ax.set_xticklabels(art_names, rotation=45)
    ax.set_ylabel("Percent")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


