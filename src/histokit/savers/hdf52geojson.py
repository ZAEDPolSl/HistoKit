

import json
import math
from pathlib import Path

import cv2
import h5py
import numpy as np
import shapely

from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, mapping, shape
from shapely.geometry.polygon import orient
from skimage.morphology import remove_small_holes, remove_small_objects
import geopandas as gpd
from ..slide import SpatialMask, BBox

try:
    from shapely import make_valid
except ImportError:
    from shapely.validation import make_valid




def repair_for_trident(features, patch_size=224):
    tolerance = patch_size / 4
    repaired_features = []

    for feature in features:
        geom = shape(feature["geometry"])

        # Simulate TRIDENT simplification
        simplified = shapely.simplify(
            geom,
            tolerance=tolerance,
            preserve_topology=True,
        )

        if not simplified.is_valid:
            print(
                "Geometry becomes invalid after "
                "TRIDENT simplification. Repairing..."
            )

            # Save a repaired version of the simplified geometry
            geom = shapely.make_valid(simplified)

            # Ensure the saved geometry also survives simplification
            check = shapely.simplify(
                geom,
                tolerance=tolerance,
                preserve_topology=True,
            )

            if not check.is_valid:
                raise ValueError(
                    "Geometry remains invalid after repair."
                )

        repaired_feature = feature.copy()
        repaired_feature["geometry"] = mapping(geom)
        repaired_features.append(repaired_feature)

    # Reproduce TRIDENT's complete operation
    simplified_geometries = [
        shapely.simplify(
            shape(feature["geometry"]),
            tolerance=tolerance,
            preserve_topology=True,
        )
        for feature in repaired_features
    ]

    merged = shapely.union_all(simplified_geometries)

    if not merged.is_valid:
        raise ValueError(
            "TRIDENT union result is invalid."
        )

    return repaired_features




def extract_polygon_parts(geometry):
    """Extract polygons from any Shapely geometry."""

    if geometry.is_empty:
        return []

    if isinstance(geometry, Polygon):
        return [geometry]

    if isinstance(geometry, (MultiPolygon, GeometryCollection)):
        polygons = []

        for part in geometry.geoms:
            polygons.extend(extract_polygon_parts(part))

        return polygons

    return []



def masks_to_geojson(
    art_mask: dict,
    output_path: str | Path,
    remove_holes: bool = False,
    include_classes: list[int] | None = None,
    min_hole: float = 0.0,
    min_area: float = 0.0,
) -> dict:
    """
    Export WSI masks to GeoJSON compatible with TRIDENT.

    Masks are not resized. Only polygon coordinates are
    transformed to WSI level 0.

    BBox format: (x0, y0, width, height).

    min_area and min_hole are expressed in mask pixels squared.

    Small objects and small holes are removed before contour extraction.
    Invalid polygon geometries are repaired using Shapely.
    """

    output_path = Path(output_path)

    width_l0, height_l0 = art_mask["level_dimensions_0"]

    mag_save = float(art_mask["mag_save"])
    mag_l0 = float(art_mask["mag_l0"])

    if (
        not math.isfinite(mag_save)
        or not math.isfinite(mag_l0)
        or mag_save <= 0
        or mag_l0 <= 0
    ):
        raise ValueError("Magnifications must be finite and positive.")

    if not math.isfinite(min_area) or min_area < 0:
        raise ValueError("min_area must be finite and non-negative.")

    if not math.isfinite(min_hole) or min_hole < 0:
        raise ValueError("min_hole must be finite and non-negative.")

    scale = mag_l0 / mag_save

    # Thresholds are expressed directly in mask pixels.
    min_area_px = math.ceil(min_area)
    min_hole_px = math.ceil(min_hole)

    valid_classes = set(range(8))

    selected_classes = (
        valid_classes.copy()
        if include_classes is None
        else set(include_classes)
    )

    if not selected_classes.issubset(valid_classes):
        raise ValueError(
            "include_classes must contain values from 0 to 7."
        )

    features = []
    tissue_id = 0

    repaired_count = 0
    skipped_count = 0

    def contour_to_ring(contour, x_offset, y_offset):
        """
        Transform local contour coordinates to WSI level 0.
        """

        points = contour.reshape(-1, 2).astype(np.float64)

        points[:, 0] = (points[:, 0] + x_offset) * scale
        points[:, 1] = (points[:, 1] + y_offset) * scale

        # Keep coordinates inside slide boundaries.
        points[:, 0] = np.clip(
            points[:, 0], 0, width_l0
        )
        points[:, 1] = np.clip(
            points[:, 1], 0, height_l0
        )

        ring = []

        for point in points.tolist():
            if not ring or ring[-1] != point:
                ring.append(point)

        if len(ring) < 3:
            return None

        if ring[0] != ring[-1]:
            ring.append(ring[0])

        if len(ring) < 4:
            return None

        if len(set(map(tuple, ring[:-1]))) < 3:
            return None

        return ring

    def extract_valid_polygons(geometry):
        """
        Recursively extract polygon components.
        """

        if geometry.is_empty:
            return []

        if isinstance(geometry, Polygon):
            return [geometry]

        if isinstance(geometry, (MultiPolygon, GeometryCollection)):
            polygons = []

            for part in geometry.geoms:
                polygons.extend(
                    extract_valid_polygons(part)
                )

            return polygons

        return []

    def repair_polygon(exterior, holes):
        """
        Create polygon and repair invalid topology if needed.

        Returns
        -------
        polygons : list[Polygon]
            Valid polygon components.
        was_repaired : bool
            Whether geometry repair was required.
        """

        try:
            polygon = Polygon(
                shell=exterior,
                holes=holes,
            )

            if polygon.is_empty:
                return [], False

            was_repaired = not polygon.is_valid

            if was_repaired:
                polygon = make_valid(polygon)

            valid_polygons = []

            for part in extract_valid_polygons(polygon):

                if part.is_empty or part.area <= 0:
                    continue

                if not part.is_valid:
                    try:
                        part = make_valid(part)
                        was_repaired = True
                    except Exception:
                        continue

                for valid_part in extract_valid_polygons(part):

                    if (
                        valid_part.is_empty
                        or not valid_part.is_valid
                        or valid_part.area <= 0
                    ):
                        continue

                    valid_polygons.append(
                        orient(valid_part, sign=1.0)
                    )

            return valid_polygons, was_repaired

        except Exception as exc:
            print(
                f"Warning: Could not repair polygon: {exc}"
            )
            return [], False

    def polygon_to_coordinates(polygon):
        """
        Convert Shapely Polygon to GeoJSON coordinates.
        """

        exterior = [
            list(point)
            for point in polygon.exterior.coords
        ]

        holes = [
            [
                list(point)
                for point in interior.coords
            ]
            for interior in polygon.interiors
        ]

        return [exterior] + holes

    for mask_data, bbox_data in zip(
        art_mask["mask"],
        art_mask["bbox"],
        strict=True,
    ):

        bbox = BBox(
            bbox_data,
            mag=mag_save,
        )

        mask = SpatialMask(
            data=np.asarray(mask_data),
            bbox=bbox,
            kind="label",
        )

        label_mask = mask.data

        x_offset = mask.bbox.x0
        y_offset = mask.bbox.y0

        for class_id in sorted(selected_classes):

            # Create boolean mask for selected class.
            binary_mask = label_mask == class_id

            if not binary_mask.any():
                continue

            # Remove small connected components.
            if min_area_px > 1:
                binary_mask = remove_small_objects(
                    binary_mask,
                    min_size=min_area_px,
                    connectivity=2,
                )

            # Fill small holes.
            if not remove_holes and min_hole_px > 1:
                binary_mask = remove_small_holes(
                    binary_mask,
                    area_threshold=min_hole_px,
                    connectivity=2,
                )

            binary_mask = np.ascontiguousarray(
                binary_mask.astype(np.uint8)
            )

            if not binary_mask.any():
                continue

            contours, hierarchy = cv2.findContours(
                binary_mask,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if hierarchy is None:
                continue

            hierarchy = hierarchy[0]

            for idx, contour in enumerate(contours):

                # Process exterior contours only.
                if hierarchy[idx][3] != -1:
                    continue

                # Skip degenerate contours.
                if len(contour) < 3:
                    skipped_count += 1
                    continue

                # Fast area check in mask pixels.
                contour_area = cv2.contourArea(contour)

                if contour_area <= 0:
                    skipped_count += 1
                    continue

                if contour_area < min_area:
                    continue

                exterior = contour_to_ring(
                    contour,
                    x_offset,
                    y_offset,
                )

                if exterior is None:
                    skipped_count += 1
                    continue

                holes = []

                if not remove_holes:

                    child_idx = hierarchy[idx][2]

                    while child_idx != -1:

                        hole_contour = contours[child_idx]

                        if len(hole_contour) >= 3:

                            hole = contour_to_ring(
                                hole_contour,
                                x_offset,
                                y_offset,
                            )

                            if hole is not None:
                                holes.append(hole)

                        child_idx = hierarchy[child_idx][0]

                # Create polygon and repair only if invalid.
                valid_polygons, was_repaired = repair_polygon(
                    exterior=exterior,
                    holes=holes,
                )

                if was_repaired:
                    repaired_count += 1

                if not valid_polygons:
                    skipped_count += 1
                    continue

                for polygon in valid_polygons:

                    # Convert area back to mask pixel units.
                    polygon_area_mask = (
                        polygon.area / (scale ** 2)
                    )

                    if polygon_area_mask < min_area:
                        continue

                    features.append({
                        "type": "Feature",
                        "properties": {
                            "tissue_id": tissue_id,
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": polygon_to_coordinates(
                                polygon
                            ),
                        },
                    })

                    tissue_id += 1


    geojson = {
        "type": "FeatureCollection",
        "name": output_path.stem,
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:EPSG::3857"
            },
        },
        "features": features,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            geojson,
            f,
            ensure_ascii=False,
        )


    print(f"GeoJSON saved: {output_path}")
    print(f"Total polygons: {len(features)}")
    print(f"Repaired polygons: {repaired_count}")
    print(f"Skipped polygons: {skipped_count}")

    return geojson


def masks_to_patch_coords(
    art_mask: dict,
    output_h5: str | Path,
    output_preview: str | Path,
    *,
    patch_size: int = 224,
    target_mag: float = 10.0,
    include_classes: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
    min_tissue_fraction: float = 0.95,
    overlap: int = 0,
    preview_max_dim: int = 2400,
    preview_max_rectangles: int = 12000,
) -> np.ndarray:
    """Generate level-0 top-left coordinates using the ORIGINAL raster masks.

    Assumptions: art_mask['bbox'] contains (x0,y0,x1,y1) at mag_save;
    art_mask['mask'] contains matching 2D label arrays; art_mask['mag_save']
    and ['mag_l0'] are available. Bounding boxes are in mask pixel units.
    This implementation assumes non-overlapping mask tiles; overlapping tiles
    can double-count foreground and must be merged before use.

    Patch footprint in level 0 = round(patch_size * mag_l0 / target_mag).
    Pixel coverage is evaluated in the mask grid, without simplifying contours.
    """
    if not (0 < min_tissue_fraction <= 1):
        raise ValueError('min_tissue_fraction must be in (0, 1].')
    if not (0 <= overlap < patch_size):
        raise ValueError('overlap must be in [0, patch_size).')
    if target_mag <= 0 or patch_size <= 0:
        raise ValueError('patch_size and target_mag must be positive.')
    if not include_classes:
        raise ValueError('Specify foreground class IDs in include_classes.')

    width, height = map(int, art_mask['level_dimensions_0'])
    mag_save = float(art_mask['mag_save'])
    mag_l0 = float(art_mask['mag_l0'])
    scale = mag_l0 / mag_save
    footprint = round(patch_size * mag_l0 / target_mag)
    step = round((patch_size - overlap) * mag_l0 / target_mag)
    if footprint < 1 or step < 1:
        raise ValueError('Patch footprint or stride rounds to zero.')
    if footprint > width or footprint > height:
        raise ValueError('Patch exceeds level-0 slide dimensions.')

    tiles = []
    for data, bbox in zip(art_mask['mask'], art_mask['bbox'], strict=True):
        labels = np.asarray(data)
        if labels.ndim != 2:
            raise ValueError('Every segmentation mask must be a 2D label array.')
        # BBox can be a sequence or a HistoKit BBox-like object.
        if hasattr(bbox, 'x0'):
            x0, y0 = float(bbox.x0), float(bbox.y0)
        else:
            x0, y0 = float(bbox[0]), float(bbox[1])
        fg = np.isin(labels, include_classes).astype(np.uint8)
        # OpenCV integral image, with a zero-padded first row and column.
        ii = cv2.integral(fg, sdepth=cv2.CV_64F)
        tiles.append((x0, y0, fg, ii))

    # Candidate coordinates only around mask tiles, snapped to the global WSI grid.
    candidates = set()
    for x0, y0, fg, _ in tiles:
        x1 = (x0 + fg.shape[1]) * scale
        y1 = (y0 + fg.shape[0]) * scale
        left = x0 * scale
        top = y0 * scale
        ix0 = max(0, math.ceil((left - footprint) / step))
        iy0 = max(0, math.ceil((top - footprint) / step))
        ix1 = min((width - footprint) // step, math.floor(x1 / step))
        iy1 = min((height - footprint) // step, math.floor(y1 / step))
        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                candidates.add((ix * step, iy * step))

    # Evaluate exact binary mask occupancy on the segmentation grid.  A patch
    # boundary can cut mask pixels; this uses center-sampling of those pixels.
    mask_patch_width = footprint / scale
    mask_patch_area = mask_patch_width ** 2
    coords = []
    for x, y in sorted(candidates, key=lambda p: (p[1], p[0])):
        mx0, my0 = x / scale, y / scale
        mx1, my1 = (x + footprint) / scale, (y + footprint) / scale
        covered = 0.0
        for tx, ty, fg, ii in tiles:
            # pixel centers in [mx0,mx1) and [my0,my1)
            a = max(0, math.ceil(mx0 - tx - 0.5))
            b = min(fg.shape[1], math.ceil(mx1 - tx - 0.5))
            c = max(0, math.ceil(my0 - ty - 0.5))
            d = min(fg.shape[0], math.ceil(my1 - ty - 0.5))
            if a < b and c < d:
                covered += ii[d, b] - ii[c, b] - ii[d, a] + ii[c, a]
        if min(1.0, covered / mask_patch_area) + 1e-12 >= min_tissue_fraction:
            coords.append((x, y))

    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 2)
    output_h5 = Path(output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as f:
        ds = f.create_dataset(
            "coords",
            data=np.asarray(coords, dtype=np.int64).reshape(-1, 2),
            compression="gzip",
        )


        ds.attrs["patch_size"] = int(patch_size)
        ds.attrs["level0_magnification"] = float(mag_l0)
        ds.attrs["target_magnification"] = float(target_mag)


        ds.attrs["patch_size_level0"] = int(footprint)
        ds.attrs["patch_level"] = 0
        ds.attrs["target_mag"] = float(target_mag)
        ds.attrs["level0_mag"] = float(mag_l0)
        ds.attrs["overlap"] = int(overlap)
        ds.attrs["min_tissue_fraction"] = float(min_tissue_fraction)
        ds.attrs["include_classes"] = np.asarray(
            include_classes, dtype=np.int32
        )
        ds.attrs["level_dim"] = np.asarray(
            [width, height], dtype=np.int64
        )

    # Preview is built from the ORIGINAL label mask, not from a slide thumbnail.
    factor = min(1.0, preview_max_dim / max(width, height))
    pw, ph = max(1, round(width * factor)), max(1, round(height * factor))
    preview = np.full((ph, pw, 3), 255, dtype=np.uint8)
    for x0, y0, fg, _ in tiles:
        h, w = fg.shape
        px0, py0 = round(x0 * scale * factor), round(y0 * scale * factor)
        px1, py1 = round((x0 + w) * scale * factor), round((y0 + h) * scale * factor)
        x_a, y_a = max(0, px0), max(0, py0)
        x_b, y_b = min(pw, px1), min(ph, py1)
        if x_b <= x_a or y_b <= y_a:
            continue
        resized = cv2.resize(fg, (max(1, px1 - px0), max(1, py1 - py0)), interpolation=cv2.INTER_NEAREST)
        part = resized[y_a-py0:y_b-py0, x_a-px0:x_b-px0].astype(bool)
        view = preview[y_a:y_b, x_a:x_b]
        view[part] = (165, 165, 165)

    # Limit drawn rectangles for readability and speed; HDF5 always has all coords.
    draw_every = max(1, math.ceil(len(coords) / preview_max_rectangles))
    for x, y in coords[::draw_every]:
        xa, ya = round(x * factor), round(y * factor)
        xb, yb = round((x + footprint) * factor), round((y + footprint) * factor)
        cv2.rectangle(preview, (xa, ya), (xb, yb), (0, 160, 0), 1)
    output_preview = Path(output_preview)
    output_preview.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_preview), preview):
        raise OSError(f'Cannot write preview to {output_preview}')
    print(f'Accepted {len(coords)} patches | footprint level0={footprint}px | stride={step}px')
    print(f'HDF5: {output_h5}\nPreview: {output_preview}')
    return coords
