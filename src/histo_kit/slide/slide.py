from typing import Optional, List
import numpy as np
from PIL import Image
from openslide import OpenSlideUnsupportedFormatError
from .backends.numpy import NumpyBackend
from .backends.openslide import OpenSlideBackend
from .backends.pil import PILBackend
from .backends.base import BaseSlideBackend
from .bbox import BBoxMode, BBox
from typing import Union, Sequence

class Slide:
    """
    Whole-slide image reader. Uses OpenSlide and PIL backends to support a wide range of formats.

    Automatically selects an appropriate backend (OpenSlide or PIL) based on
    the file format. Provides unified access to slide metadata and region
    reading at arbitrary magnifications, resolutions, or pyramid levels.

    Parameters
    ----------
    data : str
        Path to the slide file.
    mag : float, optional
        Known magnification for the slide. Overrides the value read from
        slide metadata.
    mpp : float, optional
        Microns per pixel for the slide. Overrides the value read from
        slide metadata.
    ref_mag : float, optional
        Reference magnification used for mag/mpp conversion.
        Defaults to :attr:`REF_MAG`.
    ref_mpp : float, optional
        Reference microns per pixel used for mag/mpp conversion.
        Defaults to :attr:`REF_MPP`.

    Attributes
    ----------
    REF_MAG : float
        Class-level default reference magnification (``10``).
    REF_MPP : float
        Class-level default reference microns per pixel (``1``).
    MAG_PRECISION : int
        Number of decimal places for magnification rounding (``2``).
    MPP_PRECISION : int
        Number of decimal places for MPP rounding (``4``).

    Examples
    --------
    >>> slide = Slide("path/to/slide.svs", mag=20.0)
    >>> region = slide.read_region([0, 0, 512, 512], mag=20.0)
    """

    REF_MAG = 10
    REF_MPP = 1
    MAG_PRECISION = 2
    MPP_PRECISION = 4

    def __init__(self, data: Union[str, np.ndarray, Image.Image], mag: Optional[float] = None, mpp: Optional[float] = None,
                 ref_mag: Optional[float] = None, ref_mpp: Optional[float] = None, rescale_method: Optional[Image.Resampling] = None):
        self._backend: BaseSlideBackend = self._select_backend(data)
        self._mag, self._mpp = self._resolve_scale(user_mag=mag, user_mpp=mpp)
        self._ref_mag = self.REF_MAG if ref_mag is None else ref_mag
        self._ref_mpp = self.REF_MPP if ref_mpp is None else ref_mpp
        self.rescale_method = rescale_method if rescale_method is not None else self._select_rescale_method()

    def _select_rescale_method(self) -> Image.Resampling:
        if isinstance(self._backend, OpenSlideBackend):
            return Image.Resampling.LANCZOS
        elif isinstance(self._backend, PILBackend) and self._backend.properties.get("PIL_mode", "").startswith("RGB"):
            return Image.Resampling.LANCZOS
        else:
            return Image.Resampling.NEAREST

    @staticmethod
    def _select_backend(data: str | np.ndarray | Image.Image) -> BaseSlideBackend:
        """
        Select and initialize the appropriate slide backend for the given input.

        The function chooses a backend depending on the type and format of the input:

        - If `data` is a NumPy array, it returns a `NumpyBackend`.
        - If `data` is a string (path to a slide file), it first tries to open it with `OpenSlideBackend`.
          If the file format is unsupported or cannot be opened, it falls back to `PILBackend`.
        - If `data` is already a `PIL.Image.Image`, it will use `PILBackend`.

        Parameters
        ----------
        data : str | np.ndarray | PIL.Image.Image
            Input slide, which can be:
            - a file path (str) to a slide image
            - a NumPy array representing the slide
            - a PIL Image object

        Returns
        -------
        BaseSlideBackend
            An initialized backend instance suitable for reading the input slide.

        Raises
        ------
        TypeError
            If `data` is not one of the supported types.
        """
        if isinstance(data, np.ndarray):
            return NumpyBackend(data)
        if isinstance(data, Image.Image):
            return PILBackend(data)
        if isinstance(data, str):
            try:
                return OpenSlideBackend(data)
            except (OpenSlideUnsupportedFormatError, OSError):
                return PILBackend(data) # try PIL as a fallback for unsupported formats or if OpenSlide fails to open the file
        raise TypeError(f"Unsupported type for backend selection: {type(data)}")

    def _resolve_scale(self, user_mag, user_mpp):
        """
        Resolve effective magnification and MPP from user input and backend metadata.

        User-supplied values take priority over metadata. If only one of
        ``mag`` or ``mpp`` is available, the other is derived using the
        reference constants.

        Parameters
        ----------
        user_mag : float or None
            Magnification value supplied by the caller.
        user_mpp : float or None
            Microns-per-pixel value supplied by the caller.

        Returns
        -------
        mag : float or None
            Resolved magnification.
        mpp : float or None
            Resolved microns per pixel.
        """

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
    def assoctiated_images(self) -> dict:
        """
        Associated images from the backend.

        Returns
        -------
        dict
            Key-value pairs of associated image names and PIL Image objects.
        """
        return self._backend.associated_images

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
    def level_downsamples(self) -> List[float]:
        """
        Downsample factors for each pyramid level relative to level 0.

        Returns
        -------
        list of float
            Downsample factor per level (e.g. ``[1.0, 4.0, 16.0]``).
        """
        return self._backend.level_downsamples

    @property
    def level_dimensions(self):
        """
        Dimensions of each pyramid level.

        Returns
        -------
        list of tuple of int
            ``(width, height)`` per level.
        """
        return self._backend.level_dimensions

    @property
    def level_count(self):
        """
        Number of pyramid levels in the slide.

        Returns
        -------
        int
            Total number of levels.
        """
        return self._backend.level_count

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
    def backend(self) -> BaseSlideBackend:
        """
        Underlying slide backend instance.

        Returns
        -------
        BaseSlideBackend
            Active backend (OpenSlide or PIL).
        """
        return self._backend

    @property
    def level_mag(self) -> list:
        """
        Magnification for each pyramid level.

        Computed as ``mag / downsample`` for every level. Returns a list of
        ``None`` values if the slide magnification is unknown.

        Returns
        -------
        list of float or None
            Magnification per level.
        """
        down_levels = self._backend.level_downsamples
        if self.mag is not None:
            level_mag = [round(self.mag / x, self.MAG_PRECISION) for x in down_levels]
        else:
            level_mag = [None for x in down_levels]
        return level_mag

    @property
    def level_mpp(self) -> list:
        """
        Microns per pixel for each pyramid level.

        Computed as ``mpp * downsample`` for every level. Returns a list of
        ``None`` values if the slide MPP is unknown.

        Returns
        -------
        list of float or None
            Microns per pixel per level.
        """
        down_levels = self._backend.level_downsamples
        if self.mpp is not None:
            level_mpp = [round(self.mpp * x, self.MPP_PRECISION) for x in down_levels]
        else:
            level_mpp = [None for x in down_levels]
        return level_mpp

    def get_mpp_at_mag(self, mag: float) -> Optional[float]:
        """
        Get the microns per pixel corresponding to a given magnification.

        Uses the reference constants to convert from magnification to MPP.

        Parameters
        ----------
        mag : float
            Magnification for which to compute MPP.

        Returns
        -------
        float or None
            Corresponding microns per pixel, or ``None`` if reference values are not set.
        """
        if self._ref_mag is None or self._ref_mpp is None:
            return None
        return round(self._ref_mag * self._ref_mpp / mag, self.MPP_PRECISION)

    def get_mag_at_mpp(self, mpp: float) -> Optional[float]:
        """
        Get the magnification corresponding to a given microns per pixel.

        Uses the reference constants to convert from MPP to magnification.

        Parameters
        ----------
        mpp : float
            Microns per pixel for which to compute magnification.

        Returns
        -------
        float or None
            Corresponding magnification, or ``None`` if reference values are not set.
        """
        if self._ref_mag is None or self._ref_mpp is None:
            return None
        return round(self._ref_mag * self._ref_mpp / mpp, self.MAG_PRECISION)

    def get_best_level_for_downsample(
        self,
        *,
        mag: float = None,
        mpp: float = None,
        ratio: float = None,
    ) -> int:

        """
        Return the best pyramid level index for a target resolution.

        Exactly one of ``mag``, ``mpp``, or ``ratio`` should be supplied.
        ``mag`` and ``mpp`` are converted to a downsample ratio internally.

        Parameters
        ----------
        mag : float, optional
            Target magnification.
        mpp : float, optional
            Target microns per pixel.
        ratio : float, optional
            Direct downsample ratio relative to level 0.

        Returns
        -------
        int
            Index of the best matching pyramid level. Returns ``0`` if the
            ratio cannot be determined.
        """

        if mag is not None and self.mag is not None:
            ratio = self.mag / mag

        if mpp is not None and self.mpp is not None:
            ratio = mpp / self.mpp

        if ratio is None:
            return 0

        return self._backend.get_best_level_for_downsample(ratio)

    def read_region(
        self,
        bbox: Union[BBox, Sequence[int], np.ndarray, None] = None,
        *,
        mag: float | None = None,
        mpp: float | None = None,
        level: int | None = None,
        bbox_mode: BBoxMode = BBoxMode.WH,
        color_mode: str = "RGB"
    ) -> Image.Image:

        """
        Read a rectangular region from the slide at a specified resolution.

        Exactly one of ``level``, ``mag``, or ``mpp`` must be provided to
        define the output resolution. The ``bbox`` coordinates must be given
        in the same coordinate space as the specified resolution. If ``bbox``
        is a BBox object, its magnification or mpp attributes (if set) will
        be ignored and overridden by the provided ``mag`` or ``mpp`` parameters.
        To scale regions from a different coordinate space, use the :meth:`read_object` method instead.

        Parameters
        ----------
        bbox : BBox or sequence of int/float or numpy.ndarray, optional
            Bounding box of the region to read. When BBox is not provided, the entire level will be read.
            Coordinates are assumed to be in the coordinate space defined by the provided ``mag``, ``mpp``, or pyramid level.
        mag : float, optional
            Target magnification for the output image.
        mpp : float, optional
            Target microns per pixel for the output image.
        level : int, optional
            Pyramid level index to read from.
        bbox_mode : BBoxMode, optional
            Coordinate format of ``bbox``. Defaults to ``BBoxMode.WH``.
        color_mode : str, optional
            PIL color mode for the returned image (e.g. ``"RGB"``, ``"L"``).
            Defaults to ``"RGB"``.

        Returns
        -------
        PIL.Image.Image
            The extracted image region converted to ``color_mode``.

        Raises
        ------
        ValueError
            If not exactly one of ``level``, ``mag``, ``mpp`` is provided.
        """
        if sum(x is not None for x in (level, mag, mpp)) != 1:
            raise ValueError("Provide exactly one of: level, mag, mpp")

        if bbox is None:
            # If no bbox is provided, read the entire level. We create a bbox that covers the whole level dimensions.
            bbox = BBox(0, 0, w=self.level_dimensions[0][0], h=self.level_dimensions[0][1], mag=self.mag, mpp=self.mpp)
            if mag is None and mpp is None:
               bbox = bbox.scale(mag = self.level_mag[level])
            else:
                bbox = bbox.scale(mag = mag, mpp = self.mpp)

        if mag is not None:
            bbox_norm = BBox.normalize(bbox, mode=bbox_mode, mag=mag, ref_mag=self._ref_mag, ref_mpp=self._ref_mpp)
            region = self._read_region_mag(mag, bbox_norm)
        elif mpp is not None:
            bbox_norm = BBox.normalize(bbox, mode=bbox_mode, mpp=mpp, ref_mag=self._ref_mag, ref_mpp=self._ref_mpp)
            region = self._read_region_mpp(mpp, bbox_norm)
        elif level is not None:
            bbox_norm = BBox.normalize(bbox, mode=bbox_mode, mag=self.level_mag[level], ref_mag=self._ref_mag, ref_mpp=self._ref_mpp)
            region = self._read_region_level(level, bbox_norm)
        else:
            raise ValueError("Must provide one of level, mag, or mpp to determine read_region resolution.")
        return region.convert(color_mode)

    def _read_region_level(self, level: int, bbox: BBox):
        """
        Read a region directly at a given pyramid level.

        Coordinates in ``bbox`` are assumed to be in the level's coordinate
        space. Location is converted to level-0 coordinates internally before
        passing to the backend.

        Parameters
        ----------
        level : int
            Pyramid level index.
        bbox : BBox
            Region to read, with coordinates in the target level's space.

        Returns
        -------
        PIL.Image.Image
            The raw region image at the specified level.
        """

        # Convert location to the level 0 coordinate space x0, y0
        # h and w are defined for the given level in OpenSlide
        x0, y0, w, h = bbox.as_tuple(BBoxMode.WH)
        downsample = self._backend.level_downsamples[level]
        x0_int_l0, y0_int_l0 = round(x0 * downsample), round(y0 * downsample)
        w_int = min(round(w), self.level_dimensions[level][0] - round(x0))
        h_int = min(round(h), self.level_dimensions[level][1] - round(y0))

        return self._backend.read_region(location=(x0_int_l0, y0_int_l0), level=level, size=(w_int, h_int))

    def _read_region_mag(self, mag: float, bbox: BBox):
        """
        Read a region at a requested magnification using the best pyramid level.

        Reads from the nearest higher-resolution pyramid level and rescales
        the result to match the requested magnification.

        Parameters
        ----------
        mag : float
            Target magnification for the output image.
        bbox : BBox
            Region to read, with coordinates defined at ``mag``.

        Returns
        -------
        PIL.Image.Image
            The region image rescaled to the requested magnification.
        """
        # We want to scale region from the nearest larger level to the requested magnification
        level = self.get_best_level_for_downsample(mag=mag)

        # Scale the bbox to the level's coordinate space
        bbox_level = bbox.scale(mag=self.level_mag[level])

        # Read the region at the level's resolution
        region = self._read_region_level(level, bbox_level)

        # Finally, scale the region to the requested mpp
        bbox = bbox.scale(mag=mag)
        return region.resize((round(bbox.w), round(bbox.h)), resample=self.rescale_method)

    def _read_region_mpp(self, mpp:float, bbox: BBox):
        """
        Read a region at a requested MPP using the best pyramid level.

        Reads from the nearest higher-resolution pyramid level and rescales
        the result to match the requested microns per pixel.

        Parameters
        ----------
        mpp : float
            Target microns per pixel for the output image.
        bbox : BBox
            Region to read, with coordinates defined at ``mpp``.

        Returns
        -------
        PIL.Image.Image
            The region image rescaled to the requested MPP.
        """

        # We want to scale region from the nearest larger level to the requested mpp
        level = self.get_best_level_for_downsample(mpp=mpp)

        # Scale the bbox to the level's coordinate space
        bbox_level = bbox.scale(mpp=self.level_mpp[level])

        # Read the region at the level's resolution
        region = self._read_region_level(level, bbox_level)

        # Finally, scale the region to the requested mpp
        bbox = bbox.scale(mpp=mpp)
        return region.resize((round(bbox.w), round(bbox.h)), resample=self.rescale_method)

    def read_object(
            self,
            bbox: Union[BBox, Sequence[int], np.ndarray],
            *,
            mag_bbox: float | None = None,
            mpp_bbox: float | None = None,
            mag: float | None = None,
            mpp: float | None = None,
            bbox_mode: BBoxMode = BBoxMode.WH,
            color_mode: str = "RGB"
    ) -> Image.Image:

        """
        Read a region defined in one coordinate space and output at another resolution.

        Useful when the bounding box coordinates come from annotations at a
        specific magnification/MPP that differs from the desired output
        resolution.

        Parameters
        ----------
        bbox : BBox or sequence of int/float or numpy.ndarray
            Bounding box of the object. Can be a BBox instance, a list/tuple of 4 numbers
            (x0, y0, x1, y1) or (x0, y0, w, h), or a NumPy array.
            Coordinates are assumed to be in the coordinate space defined BBox.mpp, BBox.mag or
            by `mag_bbox` or `mpp_bbox`.

        mag_bbox : float, optional
            Magnification at which the `bbox` coordinates are defined.
            - If both `mag_bbox` and `mpp_bbox` are provided, they override the values in `bbox`.
            - If only `mag_bbox` is provided, `mpp_bbox` is computed from the reference constants.
            - If neither is provided, the existing properties of `bbox` are used.
            - If `bbox` is not a BBox instance, at least one of `mag_bbox` or `mpp_bbox` must be provided to define the coordinate space.

        mpp_bbox : float, optional
            Microns per pixel at which the `bbox` coordinates are defined.
            - If both `mag_bbox` and `mpp_bbox` are provided, they override the values in `bbox`.
            - If only `mpp_bbox` is provided, `mag_bbox` is computed from the reference constants.
            - If neither is provided, the existing properties of `bbox` are used.
            - If `bbox` is not a BBox instance, at least one of `mag_bbox` or `mpp_bbox` must be provided to define the coordinate space.

        mag : float, optional
            Target magnification for the returned image. If provided, the bounding box
            will be rescaled to match this magnification.

        mpp : float, optional
            Target microns per pixel for the returned image. If provided, the bounding box
            will be rescaled to match this resolution.

        bbox_mode : BBoxMode, optional
            Format of the `bbox` coordinates. Can be `BBoxMode.WH` for (x0, y0, w, h) or
            `BBoxMode.XY` for (x0, y0, x1, y1). Defaults to `BBoxMode.WH`.

        color_mode : str, optional
            PIL color mode for the output image, e.g., `"RGB"` or `"RGBA"`. Defaults to `"RGB"`.

        Returns
        -------
        PIL.Image.Image
            The extracted object image at the requested output resolution.
        """

        if isinstance(bbox, Sequence) and mag_bbox is None and mpp_bbox is None:
            raise ValueError("For non-BBox input, at least one of mag_bbox or mpp_bbox must be provided to define the coordinate space of the bbox.")

        # Normalize bbox to BBox object
        bbox_norm = BBox.normalize(
            bbox,
            mode=bbox_mode,
            mag=mag_bbox,
            mpp=mpp_bbox,
            ref_mag=self._ref_mag,
            ref_mpp=self._ref_mpp
        )

        bbox_final = bbox_norm if mag is None and mpp is None else bbox_norm.scale(mag=mag, mpp=mpp)
        return self.read_region(bbox_final, mag=mag, mpp=mpp, bbox_mode=bbox_mode, color_mode=color_mode)
