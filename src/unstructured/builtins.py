from unstructured.dispatcher import Dispatcher

__all__ = [
    "deref",
    "shift",
    "lift",
    "cartesian",
    "compose",
    "if_",
    "minus",
    "plus",
    "mul",
    "greater",
]

builtin_dispatch = Dispatcher()


class BackendNotSelectedError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Backend not selected")


@builtin_dispatch
def deref(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def shift(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def lift(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def cartesian(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def compose(sten):
    raise BackendNotSelectedError()


@builtin_dispatch
def if_(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def minus(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def plus(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def mul(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def greater(*args):
    raise BackendNotSelectedError()
