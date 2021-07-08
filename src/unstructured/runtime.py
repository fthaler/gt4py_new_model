from typing import Union
from dataclasses import dataclass

from unstructured.builtins import BackendNotSelectedError
from unstructured.patch_helper import dispatch


@dataclass
class Offset:
    value: Union[int, str] = None


def offset(value):
    return Offset(value)


# will be monkey patched if tracing is loaded
def _fundef_impl(fun):
    return fun


def fundef(fun):
    return _fundef_impl(fun)


# will be monkey patched if tracing is loaded
def _fendef_impl(fun):
    return fun


def fendef(fun):
    return _fendef_impl(fun)


@dispatch
def closure(*args):
    return BackendNotSelectedError()
