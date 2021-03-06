# GT4Py New Semantic Model - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.  GT4Py
# New Semantic Model is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or any later version.
# See the LICENSE.txt file at the top-level directory of this distribution for
# a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from .accessors import (
    array_column_accessor,
    constant_column_accessor,
    index_column_accessor,
)


@dataclass(frozen=True)
class ArrayStorage:
    array: np.ndarray
    dimensions: Tuple[str, ...]
    origin: Tuple[int, ...]

    def __array__(self):
        return self.array

    def _k_column_accessor(self, i, j, k_size):
        return array_column_accessor(
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
    shape: Tuple[int, ...]
    dimensions: Tuple[str, ...]
    origin: Tuple[int, ...]
    index_dimension: str

    def _k_column_accessor(self, i, j, k_size):
        axis = self.dimensions.index(self.index_dimension)
        focus = self.origin[axis]
        if self.index_dimension == "i":
            focus += i
        elif self.index_dimension == "j":
            focus += j
        return index_column_accessor(
            dimension=self.index_dimension,
            focus=focus,
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


@dataclass(frozen=True)
class ConstantStorage:
    value: Any

    def _k_column_accessor(self, i, j, k_size):
        return constant_column_accessor(value=self.value, size=k_size)


def constant(value):
    return ConstantStorage(value=value)
