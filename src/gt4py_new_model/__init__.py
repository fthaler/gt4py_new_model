from importlib.metadata import version

__version__ = version(__package__)

from .dimensions import I, J, K
from .application import apply_stencil, domain, fencil, lift, stencil
from .builtins import forward, backward, polymorhic_stencil
from .storage import storage, index

__all__ = [
    "I",
    "J",
    "K",
    "apply_stencil",
    "domain",
    "fencil",
    "lift",
    "stencil",
    "polymorhic_stencil",
    "storage",
    "index",
    "forward",
    "backward",
]
