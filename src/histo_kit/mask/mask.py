import numbers
from enum import Enum
from typing import Optional, List
import numpy as np
from PIL import Image
from .backends.base import BaseMaskBackend
from .backends.numpy import NumpyMaskBackend
from typing import Union, Sequence
from ..slide.bbox import BBox, BBoxMode

class Mask:

    EXCLUDE_VALUES_DEFAULT = [0]
    REF_MAG = 10
    REF_MPP = 1
    MAG_PRECISION = 2
    MPP_PRECISION = 4

    def __init__(self,
                 mask: Union[str, np.ndarray, Sequence[np.ndarray]],
                 bbox: Optional[Union[BBox, Sequence[int], np.ndarray, Sequence[BBox]]] = None,
                 bbox_mode: BBoxMode = BBoxMode.WH,
                 mag: Optional[float] = None,
                 mpp: Optional[float] = None,
                 ref_mag: Optional[float] = None,
                 ref_mpp: Optional[float] = None,
                 rescale_method: Optional[Image.Resampling] = None,
                 exclude_values: Sequence[int | bool | Enum] = (0,)):


        self._backend: BaseMaskBackend = self._select_backend(mask)
        self._ref_mag = self.REF_MAG if ref_mag is None else ref_mag
        self._ref_mpp = self.REF_MPP if ref_mpp is None else ref_mpp
        self._mag, self._mpp = self._resolve_scale(user_mag=mag, user_mpp=mpp)
        self._bbox_mode = bbox_mode
        self._bbox_list = BBox.parse_bbox_list(bbox)

        # define bbox as entire mask if there is only one mask and no bbox provided
        if not self._bbox_list and self.mask_count == 1:
            self._bbox_list = [BBox(0, 0, w=dim[1], h=dim[0], mag=self._mag, mpp=self._mpp) for dim in self.mask_dimensions]
        elif len(self._bbox_list) != self.mask_count:
            raise ValueError(f"Number of bounding boxes ({len(self._bbox_list)}) must match number of masks ({self.mask_count})")

        self.rescale_method = rescale_method
        self._exclude_values = self._convert_exclude_values(exclude_values)



    @property
    def exclude_values(self) -> List[int]:
        """
        List of integer pixel values to exclude when applying the mask.

        Returns
        -------
        list of int
            Pixel values that should be considered as background or ignored.
        """
        return self._exclude_values

    @exclude_values.setter
    def exclude_values(self, values: Sequence[int | bool | Enum]):
        """
        Set the list of pixel values to exclude when applying the mask.

        Parameters
        ----------
        values : sequence of int, bool, or Enum
            Pixel values to exclude. Booleans will be converted to integers (False=0, True=1).
            Enums will be converted to their integer values. If None is provided, defaults to [0].
        """
        self._exclude_values = self._convert_exclude_values(values)

    def _convert_exclude_values(self, exclude_values: Sequence[int | bool | Enum]) -> list[int]:
        if exclude_values is None:
            return self.EXCLUDE_VALUES_DEFAULT # Default to excluding 0 if no values are provided
        if not isinstance(exclude_values, Sequence):
            raise TypeError("Parameter exclude_values must be a sequence")

        normalized = []

        for v in exclude_values:
            if isinstance(v, Enum):
                v = v.value
            if isinstance(v, bool):
                v = int(v)
            if not isinstance(v, numbers.Integral):
                raise TypeError(f"Exclude value {v} is not an integer, boolean, or Enum")
            normalized.append(v)
        return normalized

    @property
    def mask_values(self) -> List[int]:
        unique_values = [np.unique(m) for m in self._backend.mask_array]
        unique_values = set(np.concatenate(unique_values))
        return sorted(unique_values)

    @staticmethod
    def _select_backend(data: str | np.ndarray | Sequence[np.array]) -> BaseMaskBackend:
        if isinstance(data, np.ndarray):
            return NumpyMaskBackend(data)
        # TODO: add support for masks in h5 format
        raise TypeError(f"Unsupported type for backend selection: {type(data)}")

    def _resolve_scale(self, user_mag, user_mpp):
        mag = user_mag if user_mag is not None else self._backend.mag
        mpp = user_mpp if user_mpp is not None else self._backend.mpp

        if mag is None and mpp is not None:
            mag = self._ref_mag * self._ref_mpp / mpp

        elif mpp is None and mag is not None:
            mpp = self._ref_mag * self._ref_mpp / mag

        return mag, mpp

    @property
    def ref_mag(self) -> Optional[float]:
        """
        Reference magnification used for mag/mpp conversion.

        Returns
        -------
        float or None
            Reference magnification value.
        """
        return self._ref_mag

    @property
    def mag(self) -> Optional[float]:
        """
        Effective magnification of the slide at level 0.

        Returns
        -------
        float or None
            Magnification value, or ``None`` if it cannot be determined.
        """
        return self._mag

    @property
    def mpp(self) -> Optional[float]:
        """
        Microns per pixel of the slide at level 0.

        Returns
        -------
        float or None
            Microns per pixel value, or ``None`` if it cannot be determined.
        """
        return self._mpp


    @property
    def mask_dimensions(self):
        """
        Dimensions of each pyramid level.

        Returns
        -------
        list of tuple of int
            ``(width, height)`` per level.
        """
        return self._backend.mask_dimensions

    @property
    def mask_count(self):
        """
        Number of pyramid levels in the slide.

        Returns
        -------
        int
            Total number of levels.
        """
        return self._backend.mask_count

    @property
    def mask_array(self) -> List[np.ndarray]:
        """
        List of mask arrays for each level.

        Returns
        -------
        list of np.ndarray
            Mask arrays corresponding to each level.
        """
        return self._backend.mask_array

    @property
    def bbox_list(self) -> List[BBox]:
        """
        List of bounding boxes corresponding to each mask.

        Returns
        -------
        list of BBox
            Bounding boxes for each mask.
        """
        return self._bbox_list

    @property
    def properties(self):
        """
        Metadata properties from the backend.

        Returns
        -------
        dict
            Key-value metadata dictionary.
        """
        return self._backend.properties

    @property
    def backend(self) -> BaseMaskBackend:
        """
        Underlying slide backend instance.

        Returns
        -------
        BaseSlideBackend
            Active backend (OpenSlide or PIL).
        """
        return self._backend

    def return_rescaled(self, mag:float = None, mpp:float = None) -> "Mask":

        new_bboxes = []
        new_masks = []

        for bbox, mask in zip(self._bbox_list, self._backend.mask_array):

            bbox_scaled = bbox.scale(mag=mag, mpp=mpp)
            new_bboxes.append(bbox_scaled)

            mask_img = Image.fromarray(mask)
            mask_resized = mask_img.resize(
                (bbox_scaled.w, bbox_scaled.h),
                resample=self.rescale_method
            )

            new_masks.append(np.array(mask_resized, dtype=mask.dtype))

        return Mask(
            mask=new_masks,
            bbox=new_bboxes,
            bbox_mode=self._bbox_mode,
            mag=mag,
            mpp=mpp,
            ref_mag=self._ref_mag,
            ref_mpp=self._ref_mpp,
            rescale_method=self.rescale_method,
            exclude_values=self._exclude_values
        )


    def mask_region(self, slide, mask_idx=0, *, mag = None, mpp = None, level = None):
        region = slide.read_object(self._bbox_list[mask_idx], mag=mag, mpp=mpp, level=level)
        mask = self._backend.mask_array[mask_idx]
        mask = np.array(Image.fromarray(mask).resize(region.size, resample=self.rescale_method), dtype=bool)
        region_masked = np.array(region) * mask[..., None]
        return region_masked

    def mask_slide(self, slide, *, mag = None, mpp = None, level = None):

        region = slide.read_object(mag=mag, mpp=mpp, level=level)
        mask_all = np.zeros_like(region)

        for i in range(self.mask_count):
            bbox_i = self._bbox_list[i]
            bbox_i_scaled = bbox_i.scale(mag = self._mag, mpp = self._mpp)
            mask = self._backend.mask_array[i]
            mask = np.array(Image.fromarray(mask).resize((bbox_i_scaled.w, bbox_i_scaled.h), resample=self.rescale_method), dtype=bool)
            mask_all[bbox_i_scaled.y0:bbox_i_scaled.y1, bbox_i_scaled.x0:bbox_i_scaled.x1,:] = region[bbox_i_scaled.y0:bbox_i_scaled.y1, bbox_i_scaled.x0:bbox_i_scaled.x1,:]* mask[..., None]

        return mask_all

