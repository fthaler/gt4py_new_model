from unstructured.dispatcher import Dispatcher

__all__ = ["deref", "shift", "lift", "cartesian", "compose"]

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
