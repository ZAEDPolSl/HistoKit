from importlib import import_module

__all__ = ["gamred"]


def __getattr__(name):
    if name in __all__:
        return import_module(f".{name}", __name__)
    raise AttributeError(name)