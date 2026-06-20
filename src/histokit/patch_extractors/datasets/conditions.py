import numpy as np


class AnyExcluded:
    def __init__(self, exclude_fns):
        self.exclude_fns = exclude_fns

    def __call__(self, patch: np.ndarray) -> bool:
        return any(fn(patch) for fn in self.exclude_fns)


class ExcludeBackground:
    def __init__(self, bg_value=255, threshold=1.0):
        self.bg_value = bg_value
        self.threshold = threshold

    def __call__(self, patch: np.ndarray) -> bool:
        return np.mean(patch == self.bg_value) >= self.threshold