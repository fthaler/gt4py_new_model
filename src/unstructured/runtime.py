from typing import Union
from dataclasses import dataclass

from unstructured.builtins import BackendNotSelectedError, builtin_dispatch
from unstructured.dispatcher import Dispatcher

__all__ = ["offset", "fundef", "fendef", "closure"]

fun_fen_def_dispatch = Dispatcher()


@dataclass
class Offset:
    value: Union[int, str] = None


def offset(value):
    return Offset(value)


@fun_fen_def_dispatch
def fundef(fun):
    return BackendNotSelectedError()


@fun_fen_def_dispatch
def fendef(fun):
    return BackendNotSelectedError()


@builtin_dispatch
def closure(*args):
    return BackendNotSelectedError()
