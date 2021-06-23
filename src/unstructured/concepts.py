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

    # def __call__(self, pos):
    #     if self.consumed_location in pos.keys():
    #         new_pos = pos.copy()
    #         del new_pos[self.consumed_location]
    #         new_pos[self.new_location] = self.neighbor_table[
    #             pos[self.consumed_location]
    #         ][self.i]
    #         return new_pos
    #     return pos


@dataclass(frozen=True)
class StridedOffset:
    """
    Example

    C2E_0 = StridedOffset(remap={'IE': RelativeOffset(location='IC', offset=0), 'JE': RelativeOffset(location='JC', offset=0), 'ColorE': AbsoluteIndex(i=0)},
        consumed_locations={'IC', 'JC'}
    )
    """

    remap: dict
    consumed_locations: list

    # this is the implementation for my python embedded execution
    # def __call__(self, pos):
    #     if all(loc in pos.keys() for loc in self.consumed_locations):
    #         new_pos = pos.copy()
    #         for loc in self.consumed_locations:
    #             del new_pos[loc]

    #         for new_loc, offset in self.remap.items():
    #             new_pos[new_loc] = (
    #                 offset.i
    #                 if isinstance(offset, AbsoluteIndex)
    #                 else pos[offset.location] + offset.i
    #             )
    #         return new_pos
    #     return pos


@dataclass(frozen=True)
class OffsetGroup:  # e.g. V2E
    offsets: list

    def __call__(self, index=None):
        if index is None:
            # normal mode
            assert False
        else:
            # special mode that does shift to a concrete element of the OffsetGroup
            return self.offsets[index]
            # def impl(pos):
            #     return self.offsets[index](pos)

            # return impl
