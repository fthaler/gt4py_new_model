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


# this is super ugly
fundef_registry = {}


def fendef(*dec_args, **dec_kwargs):
    def wrapper(fun):
        def impl(*args, **kwargs):
            kwargs = {**kwargs, **dec_kwargs}

            for key, val in fundef_registry.items():
                if key is not None and key(kwargs):
                    val(fun, *args, **kwargs)
                    return
            if None in fundef_registry:
                fundef_registry[None](fun, *args, **kwargs)
                return
            raise RuntimeError("Unreachable")

        return impl

    if len(dec_args) == 1 and len(dec_kwargs) == 0 and callable(dec_args[0]):
        return wrapper(dec_args[0])
    else:
        assert len(dec_args) == 0
        return wrapper


@fun_fen_def_dispatch
def fundef(fun):
    return fun


@builtin_dispatch
def closure(*args):
    return BackendNotSelectedError()
