from dataclasses import dataclass
from typing import Union

import numpy as np

from .dimensions import Dimension
from .types import Column, Scalar


@dataclass(frozen=True)
class ArrayValueAccessor:
    array: np.ndarray
    dimensions: tuple[str, ...]
    focus: tuple[int, ...]

    def __getitem__(self, offsets: Union[Dimension, tuple[Dimension, ...]]) -> Scalar:
        if isinstance(offsets, Dimension):
            offsets = (offsets,)
        offset_dict = {o.dimension: o.offset for o in offsets}
        index = tuple(
            f + offset_dict.get(d, 0) for f, d in zip(self.focus, self.dimensions)
        )
        if self.array.ndim == 1:
            index = index[0]
        return self.array[index]


@dataclass(frozen=True)
class ArrayColumnAccessor:
    array: np.ndarray
    dimensions: tuple[str, ...]
    focus: tuple[int, ...]
    column: str
    size: int

    def __getitem__(self, offsets: Union[Dimension, tuple[Dimension, ...]]) -> Column:
        if isinstance(offsets, Dimension):
            offsets = (offsets,)
        offset_dict = {o.dimension: o.offset for o in offsets}
        index = tuple(
            slice(None) if d == self.column else f + offset_dict.get(d, 0)
            for f, d in zip(self.focus, self.dimensions)
        )
        if self.array.ndim == 1:
            index = index[0]
        res = self.array[index]
        assert res.ndim == 1
        column_focus = next(
            f for f, d in zip(self.focus, self.dimensions) if d == self.column
        )
        return np.roll(res, -column_focus - offset_dict.get(self.column, 0))[
            : self.size
        ]

    def _value_accessor(self, k):
        focus = tuple(
            f + k if d == self.column else f
            for f, d in zip(self.focus, self.dimensions)
        )
        return ArrayValueAccessor(
            array=self.array, dimensions=self.dimensions, focus=focus
        )


@dataclass(frozen=True)
class IndexValueAccessor:
    dimension: str
    focus: int
    start: int
    end: int

    def __getitem__(self, offsets: Union[Dimension, tuple[Dimension, ...]]) -> int:
        if isinstance(offsets, Dimension):
            offsets = (offsets,)
        offset = {o.dimension: o.offset for o in offsets}.get(self.dimension, 0)
        return self.focus + offset

    def on_level(self, level):
        if level < 0:
            return self.focus == self.end + level
        return self.focus == level + self.start


@dataclass(frozen=True)
class IndexColumnAccessor:
    dimension: str
    focus: int
    start: int
    end: int
    column: str
    size: int

    def __getitem__(self, offsets: Union[Dimension, tuple[Dimension, ...]]) -> Column:
        if isinstance(offsets, Dimension):
            offsets = (offsets,)
        offset = {o.dimension: o.offset for o in offsets}.get(self.dimension, 0)
        if self.column == self.dimension:
            return self.focus + offset + np.arange(self.size)
        return np.full(self.size, self.focus + offset)

    def _value_accessor(self, k):
        return IndexValueAccessor(
            dimension=self.dimension,
            focus=self.focus + k if self.dimension == "k" else self.focus,
            start=self.start,
            end=self.end,
        )
