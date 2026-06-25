from __future__ import annotations
from typing import Literal
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
import warnings
from .bbox import BBox

MaskKind = Literal["label", "prob", "binary"] 

class SpatialMask:
    """Mask array with spatial metadata.

    Exactly one of ``bbox``, ``mag``, or ``mpp`` must be provided.

    If ``bbox`` is provided, the mask uses this bbox directly. If only ``mag``
    or only ``mpp`` is provided, a bbox covering the whole mask is created in
    ``(x0, y0, w, h)`` format.

    Parameters
    ----------
    data : np.ndarray
        Input mask array. Must be 2D or 3D.
    bbox : BBox, optional
        BBox describing the mask location and coordinate space.
    kind : {"label", "probability", "binary"}, optional
        Mask type. ``"label"`` should be used for binary or class-label masks.
        ``"probability"`` should be used for confidence maps.
        ``"binary"`` is deprecated, use ``"label"`` instead.
    mag : float, optional
        Magnification of the mask coordinate space. Used only when ``bbox`` is
        not provided.
    mpp : float, optional
        Microns per pixel of the mask coordinate space. Used only when ``bbox``
        is not provided.
    ref_mag : float, optional
        Reference magnification used for ``mag``/``mpp`` conversion. Default is
        ``10.0``.
    ref_mpp : float, optional
        Reference microns per pixel used for ``mag``/``mpp`` conversion.
        Default is ``1.0``.
    """

    REF_MAG: float = 10.0
    REF_MPP: float = 1.0

    def __init__(
        self,
        data: np.ndarray,
        *,
        bbox: BBox | None = None,
        kind: MaskKind = "label",
        mag: float | None = None,
        mpp: float | None = None,
        ref_mag: float | None = None,
        ref_mpp: float | None = None,
    ):
        
        if data.ndim not in (2, 3):
            raise ValueError(
                f"Expected 2D or 3D mask, got shape {data.shape}."
            )

        if kind not in ("label", "prob", "binary"):
            raise ValueError(
                f"Unknown mask kind: {kind}. Expected 'label', 'probability', or 'binary'."
            )

        if sum(x is not None for x in (bbox, mag, mpp)) != 1:
            raise ValueError(
                "Provide exactly one of: bbox, mag, or mpp."
            )

        self.data = data
        self.kind = kind

        self.resampling_method = self._resampling_method()

        ref_mag = self.REF_MAG if ref_mag is None else ref_mag
        ref_mpp = self.REF_MPP if ref_mpp is None else ref_mpp

        if bbox is not None:
            self.bbox = bbox.get_bbox_integer()

        else:
            height, width = data.shape[:2]

            self.bbox = BBox(
                [0, 0, width, height],
                mag=mag,
                mpp=mpp,
                ref_mag=ref_mag,
                ref_mpp=ref_mpp,
            ).get_bbox_integer()

        self._validate_shape()

    @property
    def mag(self) -> float | None:
        return self.bbox.mag

    @property
    def mpp(self) -> float | None:
        return self.bbox.mpp

    @property
    def shape(self) -> tuple[int, int]:
        """Return spatial mask shape as ``(height, width)``."""
        return self.data.shape[:2]

    @property
    def size(self) -> tuple[int, int]:
        """Return spatial mask size as ``(width, height)``."""
        h, w = self.shape
        return w, h

    def resize_to_bbox(self, bbox: BBox) -> "SpatialMask":
        """Resize mask to match a target bbox.

        Parameters
        ----------
        bbox : BBox
            Target bbox. It is rasterized using ``get_bbox_integer()``.

        Returns
        -------
        SpatialMask
            New mask resized to the target bbox.
        """
        bbox_int = bbox.get_bbox_integer()

        data = self._resize_array(
            self.data,
            size=bbox_int.size,
            resample=self._resampling_method(),
        )

        return SpatialMask(
            data,
            bbox=bbox,
            kind=self.kind,
        )

    def scale(self,*,factor: float | None = None, target_mag: float | None = None, target_mpp: float | None = None) -> "SpatialMask":
        """Scale mask and bbox to another coordinate space.

        Parameters
        ----------
        factor : float, optional
            Direct scaling factor.
        target_mag : float, optional
            Target magnification.
        target_mpp : float, optional
            Target microns per pixel.

        Returns
        -------
        SpatialMask
            Scaled mask with scaled bbox.
        """
        bbox_scaled = self.bbox.scale(
            factor=factor,
            target_mag=target_mag,
            target_mpp=target_mpp,
        )

        return self.resize_to_bbox(bbox_scaled)

    def split_regions(self, min_area: int | None = None) -> list["SpatialMask"]:
        """Split a 2D label mask into connected regions.

        Parameters
        ----------
        min_area : int, optional
            Minimum connected-component area in pixels.

        Returns
        -------
        list[SpatialMask]
            Connected mask regions. Each region has its own bbox in the same
            coordinate space as the source mask.
        """
        if self.data.ndim != 2:
            raise ValueError(
                f"Expected 2D mask for splitting, got shape {self.data.shape}."
            )

        if self.kind != "label":
            raise ValueError(
                "split_regions() is only supported for kind='label'."
            )

        # label image
        labeled = label(self.data != 0, connectivity=2)
        regions: list[SpatialMask] = []

        # divide mask into regions
        for region in regionprops(labeled):
            if min_area is not None and region.area < min_area:
                continue

            min_row, min_col, max_row, max_col = region.bbox

            region_labeled = labeled[min_row:max_row, min_col:max_col]
            region_mask = self.data[min_row:max_row, min_col:max_col].copy()

            # set pixels outside the region to 0 - keep only the pixels belonging to the current region
            region_mask[region_labeled != region.label] = 0

            # region bounding box
            bbox = BBox([self.bbox.x0 + min_col, self.bbox.y0 + min_row, max_col - min_col, max_row - min_row],
                mag=self.mag,
                mpp=self.mpp,
                ref_mag=self.bbox.ref_mag,
                ref_mpp=self.bbox.ref_mpp,
            )

            # region mask
            regions.append(
                SpatialMask(
                    region_mask,
                    bbox=bbox,
                    kind=self.kind,
                )
            )
        
        # return the list of regions
        return regions 

    @classmethod
    def merge_regions(cls, regions: list["SpatialMask"], shape: tuple[int, int]) -> "SpatialMask":

        # check inputs
        if len(shape) != 2:
            raise ValueError(f"shape must be a 2-element tuple ``(height, width)``, got {shape}.")

        first = regions[0]
        output_dtype = first.data.dtype
        output_kind = first.kind 

        if first.data.ndim == 2:
            out_shape = shape
        elif first.data.ndim == 3:
            out_shape = (*shape, first.data.shape[2])
        else:
            raise ValueError(
                f"Unsupported mask shape: {first.data.shape}."
            )
        
        # initialize the output mask
        merged = np.zeros(out_shape, dtype=output_dtype)

        for region in regions:
            region._validate_shape()
            bbox = region.bbox.get_bbox_integer()

            x0, y0, x1, y1 = bbox.xyxy_int
            roi = merged[y0:y1, x0:x1]

            # add region to the output mask
            if region.data.ndim == 2:
                idx = region.data != 0
                roi[idx] = region.data[idx]

            elif region.data.ndim == 3:
                idx = np.any(region.data != 0, axis=2)
                roi[idx, :] = region.data[idx, :]

        # full bbox for the output mask
        output_bbox = BBox(
        [0, 0, shape[1], shape[0]],
        mag=first.mag,
        mpp=first.mpp,
        ref_mag=first.bbox.ref_mag,
        ref_mpp=first.bbox.ref_mpp,
        )

        return cls(
            merged,
            bbox=output_bbox,
            kind=output_kind,
        )
    
    def to_parts(self) -> tuple[np.ndarray, np.ndarray]:
        """Return mask and bbox array.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(mask, bbox)`` where bbox is ``(x0, y0, w, h)``.
        """
        return self.data, self.bbox.numpy_int()

    @staticmethod
    def parts_from_regions(regions: list["SpatialMask"]) -> tuple[list[np.ndarray], np.ndarray]:
        """Convert regions to old-style ``masks, bboxes`` representation."""
        masks = [region.data for region in regions]

        if len(regions) == 0:
            return masks, np.empty((0, 4), dtype=int)

        bboxes = np.stack(
            [
                region.bbox.numpy_int()
                for region in regions
            ]
        ).astype(int)

        return masks, bboxes
    
    
    def binarize(self, keep: int | tuple[int, ...] = 1) -> "SpatialMask":
        
        if self.data.ndim != 2:
            raise ValueError(
                f"mask must be 2-dimensional, got shape {self.data.shape}"
            )

        if self.data.dtype == bool:
            return self

        if not isinstance(keep, tuple):
            keep = (keep,)

        values = np.unique(self.data)

        valid_binary = np.all(np.isin(values, [0, 1]))
        valid_binary_uint8 = (
            self.data.dtype == np.uint8
            and np.all(np.isin(values, [0, 255]))
        )

        if not (valid_binary or valid_binary_uint8):
            warnings.warn(
                "mask is not bool, binary (0/1), or binary uint8 (0/255). "
                f"Found values: {values.tolist()}. "
                f"The mask will be converted to binary: values in {keep} "
                "will remain foreground, all other values will be set to background.",
                UserWarning,
                stacklevel=2,
            )

            self.data = np.isin(self.data, keep)
            return self

        self.data = self.data > 0
        return self
    
    def threshold(self, thr: float = 0.5) -> "SpatialMask":

        if self.data.ndim not in (2, 3):
            raise ValueError(
                f"mask must be 2- or 3-dimensional, got shape {self.data.shape}"
            )

        if self.data.dtype == bool:
            return self

        self.data = self.data > thr
        return self

    def __repr__(self) -> str:
        return (
            "SpatialMask("
            f"shape={self.data.shape}, "
            f"bbox={self.bbox}, "
            f"kind={self.kind}"
            ")"
        )
    
    def _validate_shape(self) -> None:
        if self.data.shape[:2] != self.bbox.shape:
            raise ValueError(
                f"Mask shape {self.data.shape[:2]} does not match bbox shape "
                f"{self.bbox.shape}. BBox: {self.bbox}."
            )

    def _resampling_method(self) -> Image.Resampling:
        if self.kind == "label":
            return Image.Resampling.NEAREST

        if self.kind == "prob":
            return Image.Resampling.BILINEAR

        raise ValueError(f"Unknown mask kind: {self.kind}")

    @staticmethod
    def _resize_array(
        data: np.ndarray,
        size: tuple[int, int],
        resample: Image.Resampling,
    ) -> np.ndarray:
        """Resize 2D or 3D mask array.

        Parameters
        ----------
        data : np.ndarray
            Input mask array.
        size : tuple[int, int]
            Output size as ``(width, height)``.
        resample : Image.Resampling
            PIL interpolation method.

        Returns
        -------
        np.ndarray
            Resized mask array.
        """
        if size[0] < 1 or size[1] < 1:
            raise ValueError(f"Invalid output size: {size}.")

        def resize_channel(channel: np.ndarray) -> np.ndarray:
            dtype = channel.dtype

            if np.issubdtype(dtype, np.floating):
                channel_for_pil = channel.astype(np.float32)
            else:
                channel_for_pil = channel

            resized = np.array(
                Image.fromarray(channel_for_pil).resize(size, resample=resample))

            return resized.astype(dtype, copy=False)

        if data.ndim == 2:
            return resize_channel(data)

        if data.ndim == 3:
            channels = [
                resize_channel(data[:, :, c])
                for c in range(data.shape[2])
            ]
            return np.stack(channels, axis=2).astype(data.dtype, copy=False)

        raise ValueError(f"Expected 2D or 3D mask, got shape {data.shape}.")