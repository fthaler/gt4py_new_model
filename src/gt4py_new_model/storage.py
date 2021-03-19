from dataclasses import dataclass

import numpy as np

from .accessors import ArrayColumnAccessor, IndexColumnAccessor


@dataclass(frozen=True)
class ArrayStorage:
    array: np.ndarray
    dimensions: tuple[str, ...]
    origin: tuple[int, ...]

    def __array__(self):
        return self.array

    def _k_column_accessor(self, i, j, k_size):
        return ArrayColumnAccessor(
            array=self.array,
            dimensions=self.dimensions,
            focus=tuple(
                o + i if d == "i" else o + j if d == "j" else o
                for o, d in zip(self.origin, self.dimensions)
            ),
            column="k",
            size=k_size,
        )


def storage(array, dimensions=None, origin=None):
    if dimensions is None:
        dimensions = tuple("ijkabcdefghlmnopqrstuvwxyz")[: array.ndim]
    if origin is None:
        origin = (0,) * array.ndim
    return ArrayStorage(array=np.copy(array), dimensions=dimensions, origin=origin)


@dataclass(frozen=True)
class IndexStorage:
    shape: tuple[int, ...]
    dimensions: tuple[str, ...]
    origin: tuple[int, ...]
    index_dimension: str

    def _k_column_accessor(self, i, j, k_size):
        axis = self.dimensions.index(self.index_dimension)
        focus = self.origin[axis]
        if self.index_dimension == "i":
            focus += i
        elif self.index_dimension == "j":
            focus += j
        return IndexColumnAccessor(
            dimension=self.index_dimension,
            focus=focus,
            start=self.origin[axis],
            end=self.origin[axis] + self.shape[axis],
            column="k",
            size=k_size,
        )

    def __array__(self):
        axis = self.dimensions.index(self.index_dimension)
        res = np.empty(self.shape, dtype=int)
        slices = tuple(
            slice(None) if a == axis else np.newaxis for a in range(len(self.shape))
        )
        res[...] = (self.origin[axis] + np.arange(self.shape[axis]))[slices]
        return res


def index(shape, index_dimension, dimensions=None, origin=None):
    if dimensions is None:
        dimensions = tuple("ijkabcdefghlmnopqrstuvwxyz")[: len(shape)]
    if origin is None:
        origin = (0,) * len(shape)
    return IndexStorage(
        shape=shape,
        dimensions=dimensions,
        origin=origin,
        index_dimension=index_dimension,
    )
