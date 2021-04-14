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
from numbers import Number
from typing import Optional

import numpy as np

from .dimensions import Dimension
from .accessors import accessor


@dataclass(frozen=True)
class Domain:
    i: int
    j: int
    k: int


def domain(*args):
    if len(args) == 1:
        args = args[0]
    return Domain(i=args[0], j=args[1], k=args[2])


def stencil(func):
    return func


def _unzip_accessor(acc):
    val = acc[()]
    if not isinstance(val, tuple):
        return acc

    def wrap(i):
        return accessor(lambda offs: acc[offs][i])

    return tuple(wrap(i) for i in range(len(val)))


def lift(stencil):
    def lifted(*args):
        @accessor
        def acc(offsets):
            def wrap(acc):
                return accessor(lambda offs: acc[Dimension.collect(*offsets, *offs)])

            return stencil(*(wrap(arg) for arg in args))

        return _unzip_accessor(acc)

    return lifted


def liftv(stencil):
    def lifted(*args):
        @accessor
        def acc(offsets):
            offsets_dict = {o.dimension: o.offset for o in offsets}
            k_offset = offsets_dict.pop("k", 0)
            all_but_k_offsets = tuple(Dimension(*o) for o in offsets_dict.items())

            def wrap(acc):
                return accessor(
                    lambda offs: acc[Dimension.collect(*all_but_k_offsets, *offs)]
                )

            res = stencil(*(wrap(arg) for arg in args))
            if isinstance(res, tuple):
                return tuple(np.roll(c, -k_offset) for c in res)
            return np.roll(res, -k_offset)

        return _unzip_accessor(acc)

    return lifted


def fencil(func):
    return func


def apply_stencil(stencil, domain, outputs, inputs):
    def setval(storage, i: int, j: int, k: int, value: np.ndarray):
        index = tuple(
            o + i if d == "i" else o + j if d == "j" else o + k if d == "k" else o
            for o, d in zip(storage.origin, storage.dimensions)
        )
        storage.array[index] = value[k]

    for i in range(domain.i):
        for j in range(domain.j):
            in_accessors = (inp._k_column_accessor(i, j, domain.k) for inp in inputs)
            res = stencil(*in_accessors)
            if not isinstance(res, tuple):
                res = (res,)
            for out, r in zip(outputs, res):
                for k in range(domain.k):
                    setval(out, i, j, k, r)
