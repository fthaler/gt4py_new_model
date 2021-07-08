class BackendNotSelectedError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Backend not selected")


def default_impl(*args):
    raise BackendNotSelectedError()


def _deref_impl(*args):
    raise BackendNotSelectedError()


def deref(*args):
    return _deref_impl(*args)


def _shift_impl(*args):
    raise BackendNotSelectedError()


def shift(*args):
    return _shift_impl(*args)


def _lift_impl(*args):
    raise BackendNotSelectedError()


def lift(*args):
    return _lift_impl(*args)


def apply_stencil(*args):
    raise BackendNotSelectedError()


def cartesian(*args):
    raise BackendNotSelectedError()


def compose(sten):
    raise BackendNotSelectedError()
