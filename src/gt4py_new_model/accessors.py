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

from typing import Union

import numpy as np

from .dimensions import Dimension


def accessor(func):
    class Res:
        def __getitem__(self, offsets):
            if not isinstance(offsets, tuple):
                offsets = (offsets,)
            return func(offsets)

    return Res()


def array_column_accessor(
    array: np.ndarray,
    dimensions: tuple[str, ...],
    focus: tuple[int, ...],
    column: str,
    size: int,
):
    @accessor
    def res(offsets):
        offset_dict = {o.dimension: o.offset for o in offsets}
        index = tuple(
            slice(None) if d == column else f + offset_dict.get(d, 0)
            for f, d in zip(focus, dimensions)
        )
        if array.ndim == 1:
            index = index[0]
        res = array[index]
        if not isinstance(res, np.ndarray):
            return np.array([res] * size)
        assert res.ndim == 1
        column_focus = next(f for f, d in zip(focus, dimensions) if d == column)
        return np.roll(res, -column_focus - offset_dict.get(column, 0))[:size]

    return res


def index_column_accessor(dimension: str, focus: int, column: str, size: int):
    @accessor
    def res(offsets):
        offset = {o.dimension: o.offset for o in offsets}.get(dimension, 0)
        if column == dimension:
            return np.roll(np.arange(size), -focus - offset)
        return np.full(size, focus + offset)

    return res


def constant_column_accessor(value, size: int):
    @accessor
    def res(offsets):
        return np.full(size, value)

    return res
