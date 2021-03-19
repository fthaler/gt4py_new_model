from typing import cast, overload, Union

import numpy as np

from .types import Column, ColumnAccessor, Scalar, ScanPass


def _promote(
    *args: Union[Column, Scalar]
) -> Union[tuple[Column, ...], tuple[Scalar, ...]]:
    size = None
    for arg in args:
        if isinstance(arg, Column):
            assert size is None or size == arg.size
            size = arg.size
    if not size:
        return cast(tuple[Scalar, ...], args)

    return tuple(arg if isinstance(arg, Column) else np.full(size, arg) for arg in args)


@overload
def if_then_else(condition: Column, true_value: Column, false_value: Column) -> Column:
    ...


@overload
def if_then_else(condition: Scalar, true_value: Scalar, false_value: Scalar) -> Scalar:
    ...


def if_then_else(condition, true_value, false_value):
    c, t, f = _promote(condition, true_value, false_value)
    if isinstance(c, Column):
        assert isinstance(t, Column) and isinstance(f, Column)
        res = np.empty(c.size, dtype=np.common_type(t, f))
        for i in range(c.size):
            res[i] = t[i] if c[i] else f[i]
        return res
    else:
        return t if c else f


def scan(
    func: ScanPass,
    is_forward: bool,
    init: Union[Scalar, tuple[Scalar, ...]],
    *args: ColumnAccessor
) -> Union[Column, tuple[Column, ...]]:
    ks = range(args[0].size)
    if not is_forward:
        ks = reversed(ks)
    res = []
    state = init
    for k in ks:
        state = func(state, *(arg._value_accessor(k) for arg in args))
        res.append(state)
    if not is_forward:
        res.reverse()
    if isinstance(state, tuple):
        return tuple(np.array(r) for r in zip(*res))
    return np.array(res)
