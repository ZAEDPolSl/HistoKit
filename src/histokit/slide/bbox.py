from numbers import Real
from typing import Optional, Sequence
import numpy as np
import math
from dataclasses import dataclass


class BBox:

    REF_MAG: float = 10.0
    REF_MPP: float = 1.0

    def __init__(
        self,
        bbox: Sequence[Real] | np.ndarray,
        *,
        mag: Optional[float] = None,
        mpp: Optional[float] = None,
        ref_mag: Optional[float] = None,
        ref_mpp: Optional[float] = None,
    ):
        x0, y0, w, h = self._unpack_bbox(bbox)

        self.x0 = x0
        self.y0 = y0
        self._w = w
        self._h = h

        if self._w <= 0 or self._h <= 0:
            raise ValueError(
                "Bounding box must have positive width and height."
            )

        self.ref_mag = float(ref_mag) if ref_mag is not None else self.REF_MAG
        self.ref_mpp = float(ref_mpp) if ref_mpp is not None else self.REF_MPP

        if mag is not None and mag <= 0:
            raise ValueError("mag must be positive.")

        if mpp is not None and mpp <= 0:
            raise ValueError("mpp must be positive.")

        self._mag = float(mag) if mag is not None else None
        self._mpp = float(mpp) if mpp is not None else None


    @property
    def mag(self) -> Optional[float]:
        if self._mag is None and self._mpp is not None:
            return round(self.ref_mag * self.ref_mpp / self._mpp, 2)
        return self._mag

    @property
    def mpp(self) -> Optional[float]:
        if self._mpp is None and self._mag is not None:
            return round(self.ref_mag * self.ref_mpp / self._mag, 4)
        return self._mpp
    
    @property
    def w(self) -> float:
        return self._w
    
    @property
    def h(self) -> float:
        return self._h
    
    @property
    def x1(self) -> float:
        return self.x0 + self.w
    
    @property
    def y1(self) -> float:
        return self.y0 + self.h
    
    @property
    def xywh(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0, self.w, self.h
    
    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0, self.x1, self.y1


    def scale(
            self,
            *,
            factor: Optional[float] = None,
            target_mag: Optional[float] = None,
            target_mpp: Optional[float] = None
    ) -> "BBox":

        if factor is None:
            if target_mag is not None and self.mag is not None:
                factor = target_mag / self.mag
            elif target_mpp is not None and self.mpp is not None:
                factor = self.mpp / target_mpp
            else:
                raise ValueError(
                    "Must provide either a scaling factor or target mag/mpp with known current mag/mpp."
                )

        if factor <= 0:
            raise ValueError("Scaling factor must be positive.")

        new_mag = self.mag * factor if self.mag is not None else None
        new_mpp = self.mpp / factor if self.mpp is not None else None

        return BBox(
            [self.x0 * factor,
             self.y0 * factor,
             self.w * factor,
             self.h * factor],
            mag=new_mag,
            mpp=new_mpp,
            ref_mag=self.ref_mag,
            ref_mpp=self.ref_mpp,
        )

    def area(self) -> float:
        return self.w * self.h
    
    def get_bbox_integer(self) -> "BBox":
        x0 = math.floor(self.x0)
        y0 = math.floor(self.y0)
        x1 = math.ceil(self.x1)
        y1 = math.ceil(self.y1)

        return BBox(
            [x0, y0, x1 - x0, y1 - y0],
            mag=self.mag,
            mpp=self.mpp,
            ref_mag=self.ref_mag,
            ref_mpp=self.ref_mpp,
        )
    
    @property
    def size(self) -> tuple[int, int]:
        bbox = self.get_bbox_integer()
        return int(bbox.w), int(bbox.h)
    
    @property
    def shape(self) -> tuple[int, int]:
        width, height = self.size
        return height, width
    
    @property
    def xywh_int(self) -> tuple[int, int, int, int]:
        """Return integer bbox as ``(x0, y0, w, h)``."""
        bbox = self.get_bbox_integer()
        return int(bbox.x0), int(bbox.y0), int(bbox.w), int(bbox.h)

    @property
    def xyxy_int(self) -> tuple[int, int, int, int]:
        """Return integer bbox as ``(x0, y0, x1, y1)``."""
        bbox = self.get_bbox_integer()
        return int(bbox.x0), int(bbox.y0), int(bbox.x1), int(bbox.y1)

    def numpy(
        self,
        dtype: type | np.dtype = float,
    ) -> np.ndarray:
        return np.asarray(self.xywh, dtype=dtype)

    def numpy_int(self) -> np.ndarray:
        return np.asarray(self.xywh_int, dtype=int)
    
    def __repr__(self):
        return f"BBox(x0={self.x0}, y0={self.y0}, w={self.w}, h={self.h}, mag={self.mag}, mpp={self.mpp})"
    
    @staticmethod
    def _unpack_bbox(
        bbox: Sequence[Real] | np.ndarray,
    ) -> tuple[float, float, float, float]:
        """Validate and unpack a bbox in ``(x0, y0, w, h)`` format."""
        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()

        if not isinstance(bbox, Sequence) or len(bbox) != 4:
            raise TypeError(
                "bbox must be a 4-element sequence in ``(x0, y0, w, h)`` format."
            )

        if not all(isinstance(v, Real) for v in bbox):
            raise TypeError(
                "All bbox elements must be numeric."
            )

        x0, y0, w, h = bbox

        return float(x0), float(y0), float(w), float(h)
    






