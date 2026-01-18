try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

__version__ = importlib_metadata.version(__package__)

# Import catalog API for top-level access
from xorq import catalog_api as catalog

__all__ = ["catalog", "__version__"]
