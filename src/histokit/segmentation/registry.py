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
}