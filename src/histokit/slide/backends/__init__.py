from .base import BaseSlideBackend
from .numpy import NumpyBackend
from .openslide import OpenSlideBackend
from .pil import PILBackend

__all__ = [
	"BaseSlideBackend",
	"NumpyBackend",
	"OpenSlideBackend",
	"PILBackend",
]
