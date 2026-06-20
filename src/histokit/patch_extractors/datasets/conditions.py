import numpy as np

def any_excluded(exclude_fns):
    def _fn(patch: np.ndarray) -> bool:
        return any(fn(patch) for fn in exclude_fns)
    return _fn

def exclude_background(bg_value=255, threshold=1.0):
    def _fn(patch: np.ndarray) -> bool:
        return np.mean(patch == bg_value) >= threshold
    return _fn