from typing import Optional, List
from .mask import SpatialMask
import numpy as np
from PIL import Image
from openslide import OpenSlideUnsupportedFormatError
from .backends.numpy import NumpyBackend
from .backends.openslide import OpenSlideBackend
from .backends.pil import PILBackend
from .backends.base import BaseSlideBackend
from .bbox import BBox
from typing import Union, Sequence
from os import PathLike

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

    def __init__(
        self,
        data: Union[str, PathLike, np.ndarray, Image.Image],
        mag: Optional[float] = None,
        mpp: Optional[float] = None,
        ref_mag: Optional[float] = None,
        ref_mpp: Optional[float] = None,
        rescale_method: Optional[Image.Resampling] = None,
    ):
        if isinstance(data, PathLike):
            data = str(data)

        self._backend: BaseSlideBackend = self._select_backend(data)
        self._ref_mag = self.REF_MAG if ref_mag is None else ref_mag
        self._ref_mpp = self.REF_MPP if ref_mpp is None else ref_mpp
        self._mag, self._mpp = self._resolve_scale(user_mag=mag, user_mpp=mpp)
        self.rescale_method = rescale_method if rescale_method is not None else self._select_rescale_method()

    def _select_rescale_method(self) -> Image.Resampling:
        if isinstance(self._backend, OpenSlideBackend):
            return Image.Resampling.LANCZOS
        elif isinstance(self._backend, PILBackend) and self._backend.properties.get("PIL_mode", "").startswith("RGB"):
            return Image.Resampling.LANCZOS
        else:
            return Image.Resampling.NEAREST

    @staticmethod
    def _select_backend(
        data: str | PathLike | np.ndarray | Image.Image,
    ) -> BaseSlideBackend:
        """
        Select and initialize the appropriate slide backend for the given input.

        The function chooses a backend depending on the type and format of the input:

        - If `data` is a NumPy array, it returns a `NumpyBackend`.
        - If `data` is a path-like object, it is converted to string.
        - If `data` is a string path, it first tries to open it with `OpenSlideBackend`.
        If the file format is unsupported or cannot be opened, it falls back to `PILBackend`.
        - If `data` is already a `PIL.Image.Image`, it will use `PILBackend`.

        Parameters
        ----------
        data : str | os.PathLike | np.ndarray | PIL.Image.Image
            Input slide, which can be:
            - a file path to a slide image
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
        if isinstance(data, PathLike):
            data = str(data)

        if isinstance(data, np.ndarray):
            return NumpyBackend(data)

        if isinstance(data, Image.Image):
            return PILBackend(data)

        if isinstance(data, str):
            try:
                return OpenSlideBackend(data)
            except (OpenSlideUnsupportedFormatError, OSError):
                return PILBackend(data)

        raise TypeError(
            f"Unsupported type for backend selection: {type(data)}"
        )

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
            level_mag = [None for _ in down_levels]
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
            level_mpp = [None for _ in down_levels]
        return level_mpp


    def get_size_at_mag(self, mag: float) -> Optional[tuple]:
        """
        Get the dimensions of the slide at a given magnification.

        Uses the reference constants to convert from magnification to MPP, then
        calculates the size based on the level 0 dimensions and MPP.

        Parameters
        ----------
        mag : float
            Magnification for which to compute the size.

        Returns
        -------
        tuple of int or None
            (width, height) at the specified magnification, or ``None`` if reference values are not set.
        """
        scale_factor = mag / self.mag if self.mag is not None else 1
        width = round(self.level_dimensions[0][0] * scale_factor)
        height = round(self.level_dimensions[0][1] * scale_factor)
        return (width, height)

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
        bbox: BBox | np.ndarray | Sequence[float] | Sequence[int] | None = None,
        *,
        mag: float | None = None,
        mpp: float | None = None,
        level: int | None = None,
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
        bbox : BBox or sequence of int/float or numpy.ndarray as [x0, y0, w, h], optional
            Bounding box of the region to read. When BBox is not provided, the entire level will be read.
            Coordinates are assumed to be in the coordinate space defined by the provided ``mag``, ``mpp``, or pyramid level.
        mag : float, optional
            Target magnification for the output image.
        mpp : float, optional
            Target microns per pixel for the output image.
        level : int, optional
            Pyramid level index to read from.
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
        if bbox is None:
            # If no bbox is provided, read the entire level. We create a bbox that covers the whole level dimensions.
            bbox = BBox([0, 0, self.level_dimensions[0][0], self.level_dimensions[0][1]], mag=self.mag, mpp=self.mpp)
            if mag is None and mpp is None:
               bbox = bbox.scale(target_mag = self.level_mag[level])
            else:
                bbox = bbox.scale(target_mag = mag, target_mpp = self.mpp)
        
        if not isinstance(bbox, BBox):
            bbox = BBox(bbox, mag=mag, mpp=mpp)

        if sum(x is not None for x in (level, mag, mpp)) != 1:
            raise ValueError("Provide exactly one of: level, mag, mpp")


        if mag is not None:
            bbox_scaled = bbox.scale(target_mag=mag)
            region = self._read_region_mag(mag, bbox_scaled)
        elif mpp is not None:
            bbox_scaled = bbox.scale(target_mpp=mpp)
            region = self._read_region_mpp(mpp, bbox_scaled)
        elif level is not None:
            bbox_scaled = bbox.scale(target_mag=self.level_mag[level])
            region = self._read_region_level(level, bbox_scaled)
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
        if level < 0 or level >= self.level_count:
            raise ValueError(
                f"Invalid level: {level}. Slide has {self.level_count} levels."
            )

        # get bbox as integer coordinates
        bbox_int = bbox.get_bbox_integer()
        x0, y0, w, h = bbox_int.xywh_int

        level_width, level_height = self.level_dimensions[level]

        # bbox to level bounds.
        x1 = x0 + w
        y1 = y0 + h

        x0_clip = max(0, x0)
        y0_clip = max(0, y0)
        x1_clip = min(level_width, x1)
        y1_clip = min(level_height, y1)

        w_int = x1_clip - x0_clip
        h_int = y1_clip - y0_clip

        if w_int <= 0 or h_int <= 0:
            raise ValueError(
                f"Requested bbox is outside level bounds. "
                f"bbox={bbox_int}, level={level}, "
                f"level_dimensions={self.level_dimensions[level]}"
            )

        # backends expect location in level-0 coordinates.
        downsample = self._backend.level_downsamples[level]

        x0_l0 = int(round(x0_clip * downsample))
        y0_l0 = int(round(y0_clip * downsample))

        return self._backend.read_region(
            location=(x0_l0, y0_l0),
            level=level,
            size=(w_int, h_int),
        )
    

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
        if mag <= 0:
            raise ValueError(f"Magnification must be positive. Got: {mag}")

        # Choose pyramid level used as the source.
        level = self.get_best_level_for_downsample(mag=mag)

        # Convert requested bbox from target magnification to the source level
        bbox_level = bbox.scale(target_mag=self.level_mag[level])

        # source level (nearest higher resolution) 
        region = self._read_region_level(level, bbox_level)

        # bbox as int
        bbox_out = bbox.get_bbox_integer()

        return region.resize(
            bbox_out.size,
            resample=self.rescale_method,
        )

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

        if mpp <= 0:
            raise ValueError(f"MPP must be positive. Got: {mpp}")

        if bbox.mpp is not None and abs(bbox.mpp - mpp) > 1e-6:
            raise ValueError(
                f"bbox.mpp ({bbox.mpp}) must match requested mpp ({mpp}). "
                "Scale or normalize the bbox before calling _read_region_mpp."
            )

        level = self.get_best_level_for_downsample(mpp=mpp)

        bbox_level = bbox.scale(
            target_mpp=self.level_mpp[level],
        )

        region = self._read_region_level(
            level,
            bbox_level,
        )

        bbox_out = bbox.get_bbox_integer()

        return region.resize(
            bbox_out.size,
            resample=self.rescale_method,
        )

    def read_object(
            self,
            bbox: BBox,
            *,
            mag: float | None = None,
            mpp: float | None = None,
            color_mode: str = "RGB"
    ) -> Image.Image:

        """
        Read a region defined in one coordinate space and output at another resolution.

        Useful when the bounding box coordinates come from annotations at a
        specific magnification/MPP that differs from the desired output
        resolution.

        Parameters
        ----------
        bbox : BBox
            Bounding box of the object with specified MPP or magnification.

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

        color_mode : str, optional
            PIL color mode for the output image, e.g., `"RGB"` or `"RGBA"`. Defaults to `"RGB"`.

        Returns
        -------
        PIL.Image.Image
            The extracted object image at the requested output resolution.
        """

        if bbox is None:
            raise ValueError("bbox must be provided.")

        if (mag is None) == (mpp is None):
            raise ValueError("Provide exactly one of mag or mpp.")
        
        if bbox.mag is None and bbox.mpp is None:
            raise ValueError(
                "Could not determine bbox coordinate space. Provide mag_bbox or "
                "mpp_bbox, or pass a BBox with mag or mpp metadata."
            )

        if mag is not None:
            return self.read_region(
                bbox.scale(target_mag=mag),
                mag=mag,
                color_mode=color_mode,
            )

        return self.read_region(
            bbox.scale(target_mpp=mpp),
            mpp=mpp,
            color_mode=color_mode,
        )


    def read_masked_object(
        self,
        mask: SpatialMask,
        *,
        mag: float | None = None,
        mpp: float | None = None,
        color_mode: str = "RGB",
        pad_value: tuple[int, int, int] = (255, 255, 255),
    ) -> Image.Image:


        if not isinstance(mask, SpatialMask):
                raise TypeError("mask must be a SpatialMask.")

        if (mag is None) == (mpp is None):
            raise ValueError("Provide exactly one of mag or mpp.")

        if mag is not None:
            mask_out = mask.scale(target_mag=mag)
        else:
            mask_out = mask.scale(target_mpp=mpp)

        region = self.read_region(
            mask_out.bbox,
            mag=mag,
            mpp=mpp,
            color_mode=color_mode,
        )

        mask_out.binarize()

        if mask_out.data.shape[:2] != region.size[::-1]:
            raise ValueError(
                f"Mask shape {mask_out.data.shape[:2]} does not match region "
                f"shape {region.size[::-1]}."
            )

        region_np = np.asarray(region)
        region_masked = region_np.copy()
        region_masked[~mask_out.data] = pad_value
        return Image.fromarray(region_masked).convert("RGB")

    def read_masked_slide(
            self,
            bboxes: Sequence[Union[BBox, Sequence[int], np.ndarray]],
            masks: Sequence[np.ndarray],
            *,
            mag_bbox: float | None = None,
            mpp_bbox: float | None = None,
            mag: float | None = None,
            mpp: float | None = None,
            color_mode: str = "RGB",
            max_pixels: int | None = None,
            pad_value: tuple[int, int, int] = (255, 255, 255)
    ) -> Image.Image:

        if len(bboxes) != len(masks):
            raise ValueError(
                f"bboxes and masks must have the same length, "
                f"got {len(bboxes)} and {len(masks)}"
            )

        if mag is None and mpp is None:
            raise ValueError("Provide exactly one of mag or mpp")

        if mag is not None and mpp is not None:
            raise ValueError("Provide exactly one of mag or mpp")

        masks = [self.normalize_mask(mask) for mask in masks]
        full_mask = merge_regions(masks, bboxes, self.get_full_slide_size(mag=mag_bbox, mpp=mpp_bbox))

        slide_size = self.get_full_slide_size(mag=mag, mpp=mpp)

        if max_pixels is not None:
            n_pixels = slide_size[0] * slide_size[1]
            if n_pixels > max_pixels:
                raise MemoryError("The requested slide size exceeds the maximum allowed pixels size and can be memory intensive. " \
                "Please consider using a lower magnification or increasing the max_pixels limit.")
            
        full_mask = Image.fromarray(full_mask).resize(slide_size, resample=Image.Resampling.NEAREST)
        mask_arr = np.asarray(full_mask)
        slide = np.array(self.read_region(mag=mag, mpp=mpp, color_mode=color_mode))


        if slide.ndim == 3:
            mask_arr = mask_arr[..., None]

        region_masked = slide * mask_arr
        black = np.all(region_masked == [0, 0, 0], axis=-1)
        region_masked[black] = list(pad_value)

        return Image.fromarray(region_masked.astype(slide.dtype)).convert(color_mode)

    def read_masked_objects(
        self,
        masks: Sequence[SpatialMask],
        *,
        mag: float | None = None,
        mpp: float | None = None,
        color_mode: str = "RGB",
        pad_value: tuple[int, int, int] = (255, 255, 255),
    ) -> list[Image.Image]:
        
        if not isinstance(masks, Sequence):
            raise TypeError("masks must be a sequence of SpatialMask objects.")

        if (mag is None) == (mpp is None):
            raise ValueError("Provide exactly one of mag or mpp.")

        return [self.read_masked_object(
                mask=mask,
                mag=mag,
                mpp=mpp,
                color_mode=color_mode,
                pad_value=pad_value) for mask in masks]

    def get_full_slide_size(
        self,
        *,
        mag: float | None = None,
        mpp: float | None = None,
    ) -> tuple[int, int]:
        
        """Return full slide size at the requested resolution.

        Parameters
        ----------
        mag : float, optional
            Target magnification.
        mpp : float, optional
            Target microns per pixel.

        Returns
        -------
        tuple[int, int]
            Slide size as ``(width, height)``.
        """
        if (mag is None) == (mpp is None):
            raise ValueError("Provide exactly one of mag or mpp.")

        bbox = BBox(
            [0, 0, self.level_dimensions[0][0], self.level_dimensions[0][1]],
            mag=self.mag,
            mpp=self.mpp,
            ref_mag=self._ref_mag,
            ref_mpp=self._ref_mpp,
        )

        if mag is not None:
            bbox = bbox.scale(target_mag=mag)
        else:
            bbox = bbox.scale(target_mpp=mpp)

        return bbox.size





