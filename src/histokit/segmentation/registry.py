from .tissue.he_thr.config import HeThrConfig
from .tissue.he_thr.segmenter import HeThrSegmenter

from .tissue.gamred.config import GaMRedConfig
from .tissue.gamred.segmenter import GaMRedSegmenter

from .artifact.grandqc.config import GrandQCConfig
from .artifact.grandqc.segmenter import GrandQCSegmenter


SEGMENTER_REGISTRY = {
    "gamred": {
        "config": GaMRedConfig,
        "segmenter": GaMRedSegmenter,
        "type": "tissue",
    },
    "grandqc": {
        "config": GrandQCConfig,
        "segmenter": GrandQCSegmenter,
        "type": "artifact",
    },

    "hethr": {
        "config": HeThrConfig,
        "segmenter": HeThrSegmenter,
        "type": "tissue",
    },
}