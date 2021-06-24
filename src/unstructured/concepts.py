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
from typing import Any
import numpy as np


class _Shifter:
    @classmethod
    def shift(*args):
        raise RuntimeError("shift was not properly overwritten")

    def __call__(self, *args):
        _Shifter.shift(*args)


shift = _Shifter()


def shift_impl():
    ...


def apply_stencil(stencil, offsets):
    def real_shift(*args):
        for arg in args:
            print(f"{arg}: {offsets[arg]}")

    _Shifter.shift = real_shift

    stencil()


def sten2():
    shift("test2")


def sten():
    sten2()
    shift("test")


apply_stencil(sten, {"test": 1, "test2": 5})

## Offset providers


@dataclass(frozen=True)
class AbsoluteIndex:
    i: int


@dataclass(frozen=True)
class RelativeIndex:
    location: str
    i: int


class Offset:
    ...


@dataclass(frozen=True)
class RandomAccessOffset(Offset):  # aka nth(i)
    i: int


@dataclass(frozen=True)
class NeighborTableOffset:
    i: int  # neighbor id
    neighbor_table: np.array
    new_location: Any
    consumed_location: Any


@dataclass(frozen=True)
class StridedOffset:
    """
    Example

    C2E_0 = StridedOffset(remap={'IE': RelativeOffset(location='IC', offset=0), 'JE': RelativeOffset(location='JC', offset=0), 'ColorE': AbsoluteIndex(i=0)},
        consumed_locations={'IC', 'JC'}
    )

    "I+1" = StridedOffset(remap={'I': RelativeOffset(location='I', offset=1)}, consumed_location=['I'])
    "I+0.5" = StridedOffset(remap={'IStag': RelativeOffset(location='I', offset=1)}, consumed_location=['I'])
    """

    remap: dict
    consumed_locations: list


@dataclass(frozen=True)
class OffsetGroup:  # e.g. V2E
    offsets: list

    def __call__(self, index):  # frontend feature
        return self.offsets[index]


class NeighborAxis:
    # the value in the pos dict for NeighborAxis is a list of OffsetGroups
    ...
