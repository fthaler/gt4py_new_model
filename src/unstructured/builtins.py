from unstructured.patch_helper import dispatch


class BackendNotSelectedError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Backend not selected")


def default_impl(*args):
    raise BackendNotSelectedError()


@dispatch
def deref(*args):
    raise BackendNotSelectedError()


@dispatch
def shift(*args):
    raise BackendNotSelectedError()


@dispatch
def lift(*args):
    raise BackendNotSelectedError()


@dispatch
def cartesian(*args):
    raise BackendNotSelectedError()


@dispatch
def compose(sten):
    raise BackendNotSelectedError()
