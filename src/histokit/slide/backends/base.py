from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
from PIL import Image

class BaseSlideBackend(ABC):
    """Abstract base class for whole-slide image backends.

    This abstract class defines the interface required to read different types of images.

    """
    @abstractmethod
    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
    ) -> Image.Image:
        """Read a region from the slide at the specified level.

        Parameters
        ----------
        location : Tuple[int, int]
            The (x, y) coordinates of the top-left corner of the region in
            the base level coordinate space.
        level : int
            Pyramid level to read from (0 is the base/highest-resolution
            level).
        size : Tuple[int, int]
            The (width, height) of the region to read at the requested
            ``level``.

        Returns
        -------
        PIL.Image.Image
            The extracted region as a PIL Image.
        """
        ...

    @property
    @abstractmethod
    def level_downsamples(self) -> List[float]:
        """Downsample factor for each pyramid level.

        Returns
        -------
        list of float
            A list where the i-th element is the downsample factor of level
            ``i`` relative to the base level. Level 0 has a downsample ratio = 1.0.
        """
        ...


    @property
    @abstractmethod
    def level_dimensions(self) -> List[Tuple[int, int]]:
        """Pixel dimensions for each pyramid level.

        Returns
        -------
        list of tuple of int
            A list of (width, height) tuples for each level in the slide
            pyramid.
        """
        ...

    @property
    def associated_images(self):
        """Dictionary of associated images (e.g., thumbnails).

        Returns
        -------
        dict
            A mapping of associated image names to image data/objects. By
            default returns an empty dict.
        """

        return {}

    @property
    @abstractmethod
    def level_count(self) -> int:
        """Number of pyramid levels available in the slide.

        Returns
        -------
        int
            Total number of levels in the WSI pyramid.
        """
        ...

    @property
    @abstractmethod
    def properties(self) -> dict:
        """Slide-level properties and metadata.

        Returns
        -------
        dict
            A dictionary with backend-specific slide metadata (e.g., vendor
            tags, acquisition info).
        """
        ...

    @property
    @abstractmethod
    def mag(self) -> Optional[float]:
        """Nominal magnification of the slide, if available.

        Returns
        -------
        float or None
            The objective magnification (e.g., 20.0, 40.0) or ``None`` when
            the value is unknown.
        """
        ...

    @property
    @abstractmethod
    def mpp(self) -> Optional[float]:
        """Microns-per-pixel (MPP) of the base level, if available.

        Returns
        -------
        float or None
            The physical size of a pixel in micrometers at the base level,
            or ``None`` when not provided by the backend.
        """
        ...

    def get_best_level_for_downsample(self, ratio: float) -> int:
        """Return the level that best matches the requested downsample ratio.

        Parameters
        ----------
        ratio : float
            Requested downsample ratio relative to the base level.

        Returns
        -------
        int
            Index of the pyramid level whose downsample factor is closest to
            ``ratio``.
        """
        diffs = [abs(ds - ratio) for ds in self.level_downsamples]
        return diffs.index(min(diffs))

