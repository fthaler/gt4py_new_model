from collections.abc import Mapping
from numbers import Number
from typing import Optional, overload, Protocol, Union

import numpy as np

from .dimensions import Dimension

Scalar = Number
Column = np.ndarray
Value = Union[Column, Scalar]
Offset = Union[Dimension, tuple[Dimension]]


class ScalarAccessor(Protocol):
    def __getitem__(self, offset: Offset) -> Scalar:
        ...


class ColumnAccessor(Protocol):
    def __getitem__(self, offset: Offset) -> Column:
        ...


Accessor = Union[ScalarAccessor, ColumnAccessor]


class Stencil(Protocol):
    @overload
    def __call__(self, *args: ColumnAccessor) -> Union[Column, tuple[Column, ...]]:
        ...

    @overload
    def __call__(self, *args: ScalarAccessor) -> Union[Scalar, tuple[Scalar, ...]]:
        ...

    def __call__(
        self, *args: Accessor
    ) -> Union[Value, tuple[Column, ...], tuple[Scalar, ...]]:
        ...


class ScanPass(Protocol):
    def __call__(
        self, previous: Union[Scalar, tuple[Scalar, ...]], *args: Scalar
    ) -> Union[Scalar, tuple[Scalar, ...]]:
        ...


def column(x) -> Column:
    return np.array(x)
