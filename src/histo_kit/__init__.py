
__all__ = [
    'utils',
    'tissue_seg',
    'stain_normalisation',
    'mil',
    'grand_qc',
    'foundation_models'
]

def __getattr__(name):
    if name in __all__:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(name)