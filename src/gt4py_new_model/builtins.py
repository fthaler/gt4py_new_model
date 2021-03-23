import numpy as np

from .accessors import accessor


def _is_column_accessor(acc):
    return isinstance(acc[()], np.ndarray)


def _has_column_accessors(accs):
    return any(_is_column_accessor(acc) for acc in accs)


def _column_size(accs):
    sizes = tuple(acc[()].size for acc in accs if _is_column_accessor(acc))
    assert len(sizes) > 0
    res = sizes[0]
    for size in sizes:
        assert size == res
    return res


def _demote_accessor(acc, k):
    return accessor(lambda offs: acc[offs][k]) if _is_column_accessor(acc) else acc


def _unzip_column(col):
    return (
        tuple(np.array([x[i] for x in col]) for i in range(len(col[0])))
        if isinstance(col[0], tuple)
        else col
    )


def polymorphic_stencil(func):
    def wrapper(*accs):
        if not _has_column_accessors(accs):
            return func(*accs)

        return _unzip_column(
            np.array(
                [
                    func(*(_demote_accessor(acc, k) for acc in accs))
                    for k in range(_column_size(accs))
                ]
            )
        )

    return wrapper


def _element_accessor(acc, k):
    return accessor(lambda offs: acc[offs][k])


def scan(func, is_forward, init):
    def wrapper(*args):
        ks = range(args[0][()].size)
        if not is_forward:
            ks = reversed(ks)
        res = []
        state = init
        for k in ks:
            state = func(state, *(_element_accessor(arg, k) for arg in args))
            res.append(state)
        if not is_forward:
            res.reverse()
        return (
            tuple(np.array(r) for r in zip(*res))
            if isinstance(state, tuple)
            else np.array(res)
        )

    return wrapper


def _decorator(is_forward):
    def decorator(*args, init=None):
        if args:
            assert len(args) == 1
            assert init == None
            return scan(args[0], is_forward=is_forward, init=None)

        def wrapper(func):
            return scan(func, is_forward=is_forward, init=init)

        return wrapper

    return decorator


forward = _decorator(True)
backward = _decorator(False)
