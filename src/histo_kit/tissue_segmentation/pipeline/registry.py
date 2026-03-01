from typing import Dict, Type

SEGMENTERS: Dict[str, Type] = {}

def register_segmenter(name):
    def decorator(cls):
        SEGMENTERS[name] = cls
        return cls
    return decorator

SAVERS: Dict[str, Type] = {}

def register_saver(name: str):
    def decorator(cls):
        SAVERS[name] = cls
        return cls
    return decorator