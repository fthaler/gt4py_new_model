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


@dataclass(frozen=True)
class Dimension:
    dimension: str
    offset: int = 0

    def __add__(self, other):
        if isinstance(other, Dimension):
            assert other.dimension == self.dimension
            return Dimension(
                dimension=self.dimension, offset=self.offset + other.offset
            )
        return Dimension(dimension=self.dimension, offset=self.offset + other)

    def __sub__(self, other):
        if isinstance(other, Dimension):
            assert other.dimension == self.dimension
            return Dimension(
                dimension=self.dimension, offset=self.offset - other.offset
            )
        return Dimension(dimension=self.dimension, offset=self.offset - other)

    @staticmethod
    def collect(*indices: "Dimension") -> tuple["Dimension"]:
        res = dict[str, int]()
        for i in indices:
            res[i.dimension] = i.offset + res.get(i.dimension, 0)
        return tuple(Dimension(dimension=k, offset=v) for k, v in res.items())


I = Dimension("i")
J = Dimension("j")
K = Dimension("k")
