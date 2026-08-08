import os
import re


def parse_grid_search_params(path):
    folder_name = None

    for part in path.split(os.sep):
        if part.startswith("blending_mode_"):
            folder_name = part
            break

    if folder_name is None:
        raise ValueError(f"Cannot find grid-search folder in path: {path}")

    result = {
        "mode_overlap": "",
        "overlap": "",
        "sigma": "",
    }

    mode_match = re.search(r"blending_mode_([^_]+)", folder_name)
    overlap_match = re.search(r"overlap_([0-9]+p[0-9]+|[0-9]+)", folder_name)
    sigma_match = re.search(r"blending_sigma_([0-9]+p[0-9]+|[0-9]+)", folder_name)

    if mode_match:
        result["mode_overlap"] = mode_match.group(1)

    if overlap_match:
        result["overlap"] = float(overlap_match.group(1).replace("p", "."))

    if sigma_match:
        result["sigma"] = float(sigma_match.group(1).replace("p", "."))

    return result

path = "/mnt/warehouse/Projects/HE/Data/Artifacts Segmentation/TCGA_CompassNMD/Results/Histokit_30_06_2026/grid_search/blending_mode_constant__overlap_0p5/artifact_detection/grandqc/masks_cropped_color"

params = parse_grid_search_params(path)
print(params)