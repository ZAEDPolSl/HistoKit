from enum import Enum
from typing import Optional, Union, Sequence, List
import numpy as np

class BBoxMode(Enum):
    """
    Enumeration for bounding box coordinate formats.

    Attributes
    ----------
    XY : str
        Coordinates of upper left (x0, y0) and bottom right (x1, y1) corners.
    WH : str
        Coordinates of upper left corner (x0, y0) and width/height (w, h).
    """
    XY = "xy"
    WH = "wh"

class BBox:
    """
        Axis-aligned bounding box with optional magnification and resolution metadata.

        Both corner-based (x0, y0, x1, y1) and corner+size-based (x0, y0, w, h) initializations are supported,
        but they are mutually exclusive. The class also supports automatic conversion between magnification and
        microns per pixel based on provided reference values.

        Parameters
        ----------
        x0 : float
            Left coordinate of the bounding box.
        y0 : float
            Top coordinate of the bounding box.
        x1 : float, optional
            Right coordinate. Mutually exclusive with ``w``.
        y1 : float, optional
            Bottom coordinate. Mutually exclusive with ``h``.
        w : float, optional
            Width of the bounding box. Mutually exclusive with ``x1``.
        h : float, optional
            Height of the bounding box. Mutually exclusive with ``y1``.
        mag : float, optional
            Magnification at which the coordinates are defined.
        mpp : float, optional
            Microns per pixel at which the coordinates are defined.
        ref_mag : float, optional
            Reference magnification used for mag/mpp conversion.
            Defaults to :attr:`REF_MAG`.
        ref_mpp : float, optional
            Reference microns per pixel used for mag/mpp conversion.
            Defaults to :attr:`REF_MPP`.

        Attributes
        ----------
        REF_MAG : float
            Class-level default reference magnification (``10.0``).
        REF_MPP : float
            Class-level default reference microns per pixel (``1.0``).
        x0, y0, x1, y1 : float
            Corner coordinates of the bounding box.
        w, h : float
            Width and height of the bounding box.
        ref_mag, ref_mpp : float
            Reference values used for magnification/MPP conversions.

        Raises
        ------
        ValueError
            If both ``x1`` and ``w`` (or ``y1`` and ``h``) are provided simultaneously,
            if neither pair is provided, or if the resulting width or height is non-positive.

        Examples
        --------
        >>> bbox = BBox(10, 20, w=100, h=50, mag=20.0)
        >>> bbox.x1
        110
        >>> bbox.mpp  # derived from mag using ref values
        0.5
        """

    REF_MAG: float = 10.0
    REF_MPP: float = 1.0

    def __init__(
        self,
        x0: float,
        y0: float,
        *,
        x1: Optional[float] = None,
        y1: Optional[float] = None,
        w: Optional[float] = None,
        h: Optional[float] = None,
        mag: Optional[float] = None,
        mpp: Optional[float] = None,
        ref_mag: Optional[float] = None,
        ref_mpp: Optional[float] = None,
    ):
        self.x0 = x0
        self.y0 = y0

        if (x1 is not None and w is not None) or (y1 is not None and h is not None):
            raise ValueError("Provide either x1/y1 OR w/h (width and height), not both.")
        if (x1 is None and w is None) or (y1 is None and h is None):
            raise ValueError("Must provide either x1/y1 or w/h (width and height).")

        self.x1 = x1 if x1 is not None else x0 + w
        self.y1 = y1 if y1 is not None else y0 + h

        self.w = self.x1 - self.x0
        self.h = self.y1 - self.y0
        if self.w <= 0 or self.h <= 0:
            raise ValueError("Bounding box must have positive width and height.")

        self.ref_mag = ref_mag if ref_mag is not None else self.REF_MAG
        self.ref_mpp = ref_mpp if ref_mpp is not None else self.REF_MPP

        self._mag = mag
        self._mpp = mpp

    @property
    def mag(self) -> Optional[float]:
        """
        Magnification for which bounding box coordinates are defined.

        If ``mag`` was not provided at construction time but ``mpp`` is known,
        magnification is derived as ``ref_mag * ref_mpp / mpp``.

        Returns
        -------
        float or None
            Magnification value, or ``None`` if it cannot be determined.
        """
        if self._mag is None and self._mpp is not None:
            return round(self.ref_mag * self.ref_mpp / self._mpp, 2)
        return self._mag

    @property
    def mpp(self) -> Optional[float]:
        """
        Microns per pixel for which bounding box coordinates are defined.

        If ``mpp`` was not provided at construction time but ``mag`` is known,
        MPP is derived as ``ref_mag * ref_mpp / mag``.

        Returns
        -------
        float or None
            Microns per pixel value, or ``None`` if it cannot be determined.
        """
        if self._mpp is None and self._mag is not None:
            return round(self.ref_mag * self.ref_mpp / self._mag, 4)
        return self._mpp

    @property
    def center(self) -> tuple[float, float]:
        """
        Center point of the bounding box.

        Returns
        -------
        tuple of float
            ``(cx, cy)`` coordinates of the bounding box center.
        """
        return self.x0 + self.w / 2, self.y0 + self.h / 2

    def scale(
            self,
            *,
            factor: Optional[float] = None,
            mag: Optional[float] = None,
            mpp: Optional[float] = None
    ) -> "BBox":
        """
        Return a new bounding box scaled by a factor or to match a target resolution.

        Exactly one of ``factor``, ``mag``, or ``mpp`` must be supplied (or a
        combination that allows ``factor`` to be derived). Priority order:

        1. ``factor`` — used directly if provided.
        2. ``mag`` — factor computed as ``mag / self.mag`` (requires known ``self.mag``).
        3. ``mpp`` — factor computed as ``self.mpp / mpp`` (requires known ``self.mpp``).

        Parameters
        ----------
        factor : float, optional
            Direct scaling factor. Values > 1 enlarge, values in (0, 1) shrink.
        mag : float, optional
            Target magnification. The bounding box is rescaled so that its
            coordinates correspond to this magnification level.
        mpp : float, optional
            Target microns per pixel. The bounding box is rescaled accordingly.

        Returns
        -------
        BBox
            New :class:`BBox` instance with scaled coordinates and updated
            ``mag``/``mpp`` metadata.

        Raises
        ------
        ValueError
            If ``factor`` cannot be determined from the supplied arguments, or
            if the resulting factor is non-positive.

        Examples
        --------
        >>> bbox = BBox(0, 0, w=100, h=50, mag=10.0)
        >>> scaled = bbox.scale(mag=20.0)
        >>> scaled.w
        200.0
        """

        if factor is None:
            if mag is not None and self.mag is not None:
                factor = mag / self.mag
            elif mpp is not None and self.mpp is not None:
                factor = self.mpp / mpp
            else:
                raise ValueError(
                    "Must provide either a scaling factor or target mag/mpp with known current mag/mpp."
                )
        else:
            if factor <= 0:
                raise ValueError("Scaling factor must be positive.")

        new_mag = self.mag * factor if self.mag is not None else None
        new_mpp = self.mpp / factor if self.mpp is not None else None

        return BBox(
            x0=self.x0 * factor,
            y0=self.y0 * factor,
            w=self.w * factor,
            h=self.h * factor,
            mag=new_mag,
            mpp=new_mpp,
            ref_mag=self.ref_mag,
            ref_mpp=self.ref_mpp,
        )

    def as_tuple(self, mode: BBoxMode = BBoxMode.WH) -> tuple[float, float, float, float]:
        """
        Return the bounding box as a plain tuple.

        Parameters
        ----------
        mode : BBoxMode, optional
            Output format. ``BBoxMode.WH`` (default) returns ``(x0, y0, w, h)``;
            ``BBoxMode.XY`` returns ``(x0, y0, x1, y1)``.

        Returns
        -------
        tuple of float
            Four-element tuple representing the bounding box.

        Raises
        ------
        ValueError
            If an unsupported ``BBoxMode`` is given.
        """
        if mode == BBoxMode.XY:
            return self.x0, self.y0, self.x1, self.y1
        elif mode == BBoxMode.WH:
            return self.x0, self.y0, self.w, self.h
        else:
            raise ValueError("Invalid mode. Use 'BBoxMode.XY' or 'BBoxMode.WH'.")

    def as_array(self, mode: BBoxMode = BBoxMode.XY) -> np.ndarray:
        """
        Return the bounding box as a NumPy array.

        Parameters
        ----------
        mode : BBoxMode, optional
            Output format passed to :meth:`as_tuple`. Defaults to ``BBoxMode.XY``.

        Returns
        -------
        numpy.ndarray
            1-D float array of shape ``(4,)``.
        """
        return np.array(self.as_tuple(mode), dtype=float)

    def area(self) -> float:
        """
        Compute the area of the bounding box.

        Returns
        -------
        float
            Area of the bounding box (``w * h``).
        """
        return self.w * self.h

    @classmethod
    def normalize(
            cls,
            bbox: Union["BBox", Sequence[int], np.ndarray],
            mode: BBoxMode = BBoxMode.WH,
            mag: Optional[float] = None,
            mpp: Optional[float] = None,
            ref_mag: Optional[float] = None,
            ref_mpp: Optional[float] = None,
    ) -> "BBox":
        """
        Coerce various input types into a :class:`BBox` instance.

        Accepts an existing :class:`BBox`, a length-4 numeric sequence, or a
        ``numpy.ndarray``. When an existing :class:`BBox` is passed together
        with new ``mag``/``mpp``/``ref_mag``/``ref_mpp`` values, a new
        :class:`BBox` is returned with the updated metadata.

        Parameters
        ----------
        bbox : BBox or sequence of int/float or numpy.ndarray
            Input bounding box representation.
        mode : BBoxMode, optional
            Coordinate format of the input when ``bbox`` is a sequence or array.
            Defaults to ``BBoxMode.WH``.
        mag : float, optional
            Magnification to assign to the resulting :class:`BBox`.
        mpp : float, optional
            Microns per pixel to assign to the resulting :class:`BBox`.
        ref_mag : float, optional
            Reference magnification for conversion. If not None, overrides the value stored
            in ``bbox`` when ``bbox`` is a :class:`BBox`.
        ref_mpp : float, optional
            Reference microns per pixel. If not None, overrides the value stored in ``bbox``
            when ``bbox`` is a :class:`BBox`.

        Returns
        -------
        BBox
            Normalised :class:`BBox` instance.

        Raises
        ------
        TypeError
            If ``bbox`` is not a :class:`BBox`, sequence, or ``numpy.ndarray``,
            or if a sequence contains non-numeric elements.
        ValueError
            If a sequence does not have exactly 4 elements, or if an invalid
            ``BBoxMode`` is provided.

        Examples
        --------
        >>> BBox.normalize([10, 20, 50, 30], mode=BBoxMode.WH, mag=20.0)
        BBox(x0=10, y0=20, w=50, h=30, mag=20.0)
        """
        if isinstance(bbox, cls):
            if any(x is not None for x in (mag, mpp, ref_mag, ref_mpp)):

                recalculate = (mag is not None or mpp is not None)

                return cls(
                    bbox.x0,
                    bbox.y0,
                    w=bbox.w,
                    h=bbox.h,
                    mag=mag if recalculate else bbox.mag,
                    mpp=mpp if recalculate else bbox.mpp,
                    ref_mag=ref_mag if ref_mag is not None else bbox.ref_mag,
                    ref_mpp=ref_mpp if ref_mpp is not None else bbox.ref_mpp,
                )

            return bbox

        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()

        if isinstance(bbox, Sequence):
            if not all(isinstance(x, (int, float)) for x in bbox):
                raise TypeError("All elements of bbox sequence must be numeric (int or float)")
            if len(bbox) != 4:
                raise ValueError("bbox must have 4 elements")

            x0, y0, a, b = bbox

            if mode == BBoxMode.XY:
                return cls(x0, y0, x1=a, y1=b, mag=mag, mpp=mpp, ref_mag=ref_mag, ref_mpp=ref_mpp)
            elif mode == BBoxMode.WH:
                return cls(x0, y0, w=a, h=b, mag=mag, mpp=mpp, ref_mag=ref_mag, ref_mpp=ref_mpp)

            raise ValueError("Invalid BBoxMode")

        raise TypeError("bbox must be BBox or sequence of 4 numbers")

    @staticmethod
    def parse_bbox_list(
        bbox: Optional[Union['BBox', Sequence[int], np.ndarray, Sequence['BBox']]],
        bbox_mode: 'BBoxMode' = BBoxMode.WH
    ) -> List['BBox']:
        """
        Converts various bbox formats into a list of BBox objects.

        Parameters
        ----------
        bbox : BBox | Sequence[int] | np.ndarray | Sequence[BBox] | None
            Input bounding box or list/array of boxes.
        bbox_mode : BBoxMode
            Mode of bounding box (WH or XY).

        Returns
        -------
        List[BBox] or None
        """
        if bbox is None:
            return []

        # Single BBox
        if isinstance(bbox, BBox):
            return [bbox]


        if isinstance(bbox, np.ndarray):
            # NumPy array (4, )
            if bbox.ndim == 1 and bbox.size == 4:
                x, y, a, b = tuple(bbox)
                if bbox_mode == BBoxMode.WH:
                    return [BBox(x, y, w=a, h=b)]
                else:
                    return [BBox(x, y, x1=a, y1=b)]
            # NumPy array (N, 4)
            elif bbox.ndim == 2 and bbox.shape[1] == 4:
                bboxes = []
                for row in bbox:
                    x, y, a, b = tuple(row)
                    if bbox_mode == BBoxMode.WH:
                        bboxes.append(BBox(x, y, w=a, h=b))
                    else:
                        bboxes.append(BBox(x, y, x1=a, y1=b))
                return bboxes
            else:
                raise ValueError(f"bbox array must be 1D with 4 elements or 2D with shape (N,4), got {bbox.shape}")

        # Sequence of BBoxes or 4-element sequences
        if isinstance(bbox, Sequence):
            bboxes = []
            for i, b in enumerate(bbox):
                if isinstance(b, BBox):
                    bboxes.append(b)
                elif isinstance(b, Sequence) and len(b) == 4:
                    x, y, a, b_ = b
                    if bbox_mode == BBoxMode.WH:
                        bboxes.append(BBox(x, y, w=a, h=b_))
                    else:
                        bboxes.append(BBox(x, y, x1=a, y1=b_))
                else:
                    raise TypeError(f"Invalid bbox at index {i}: must be BBox or sequence of 4 ints")
            return bboxes

        raise TypeError(f"Invalid type for bbox: {type(bbox)}")



