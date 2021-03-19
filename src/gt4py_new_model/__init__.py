from importlib.metadata import version

__version__ = version(__package__)

from .dimensions import I, J, K
from .application import apply_stencil, domain, fencil, lift, scaniter, stencil
from .builtins import if_then_else, scan
from .storage import storage, index

__all__ = [
    "I",
    "J",
    "K",
    "apply_stencil",
    "domain",
    "fencil",
    "lift",
    "scaniter",
    "stencil",
    "if_then_else",
    "scan",
    "storage",
    "index",
]
