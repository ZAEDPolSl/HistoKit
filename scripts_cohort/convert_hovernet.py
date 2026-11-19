
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# Dostosuj kolory do swojego modelu.
# Format: class_id: (class_name, RGB)
CLASS_INFO = {
    0: ("Type 0", (150, 150, 150)),
    1: ("Type 1", (255, 0, 0)),
    2: ("Type 2", (0, 255, 0)),
    3: ("Type 3", (0, 0, 255)),
    4: ("Type 4", (255, 255, 0)),
    5: ("Type 5", (255, 0, 255)),
}


def rgb_to_int(rgb):
    r, g, b = rgb
    return (r << 16) | (g << 8) | b


def convert(input_path, output_path, png_path, downsample=8):
    if downsample < 1:
        raise ValueError("downsample must be >= 1")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nuclei = data["nuc"]

    features = []
    polygons = []
    skipped = 0

    # Collect valid polygons and determine their bounds.
    for nucleus_id, nucleus in nuclei.items():
        contour = np.asarray(
            nucleus.get("contour", []),
            dtype=np.float64
        )

        if (
            contour.ndim != 2
            or contour.shape[1] != 2
            or len(contour) < 3
            or not np.isfinite(contour).all()
        ):
            skipped += 1
            continue

        cell_type = int(nucleus.get("type", 0))

        class_name, color = CLASS_INFO.get(
            cell_type,
            (f"Type {cell_type}", (180, 180, 180))
        )

        # GeoJSON: original WSI coordinates
        coordinates = contour.tolist()

        if coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coordinates]
            },
            "properties": {
                "objectType": "annotation",
                "classification": {
                    "name": class_name,
                    "colorRGB": rgb_to_int(color)
                },
                # No individual nucleus name/label.
                "isLocked": False
            }
        }

        features.append(feature)
        polygons.append((contour, color))

    # Save QuPath GeoJSON
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            geojson,
            f,
            ensure_ascii=False,
            separators=(",", ":")
        )

    # Generate downsampled segmentation PNG
    if not polygons:
        raise ValueError("No valid nuclei contours found.")

    all_min = np.min(
        [np.min(p, axis=0) for p, _ in polygons],
        axis=0
    )
    all_max = np.max(
        [np.max(p, axis=0) for p, _ in polygons],
        axis=0
    )

    # Crop visualization to the area containing nuclei.
    x0 = int(np.floor(all_min[0]))
    y0 = int(np.floor(all_min[1]))
    x1 = int(np.ceil(all_max[0])) + 1
    y1 = int(np.ceil(all_max[1])) + 1

    width = int(np.ceil((x1 - x0) / downsample))
    height = int(np.ceil((y1 - y0) / downsample))

    # OpenCV uses BGR color order.
    mask = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    for contour, rgb in polygons:
        scaled = np.rint(
            (contour - np.array([x0, y0])) / downsample
        ).astype(np.int32)

        bgr = (rgb[2], rgb[1], rgb[0])

        cv2.fillPoly(
            mask,
            [scaled.reshape(-1, 1, 2)],
            color=bgr
        )

    png_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(png_path), mask):
        raise RuntimeError(f"Could not save PNG: {png_path}")

    print(f"Input nuclei: {len(nuclei):,}")
    print(f"Converted annotations: {len(features):,}")
    print(f"Skipped invalid contours: {skipped:,}")
    print(f"GeoJSON saved: {output_path}")
    print(f"PNG saved: {png_path}")
    print(f"PNG size: {width} x {height}")
    print(f"Downsample: {downsample}x")
    print(f"PNG crop origin (WSI pixels): ({x0}, {y0})")


def main():
    parser = argparse.ArgumentParser(
        description="Convert nuclei JSON to QuPath GeoJSON and PNG"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default="C:/Users/julia/Desktop/SS45212_R0A10F22_232337.json",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default="C:/Users/julia/Desktop/SS45212_R0A10F22_232337.geojson",
    )

    parser.add_argument(
        "--png",
        type=Path,
        default="C:/Users/julia/Desktop/SS45212_R0A10F22_232337.png",
    )

    parser.add_argument(
        "--downsample",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    convert(
        input_path=args.input,
        output_path=args.output,
        png_path=args.png,
        downsample=args.downsample
    )


if __name__ == "__main__":
    main()


