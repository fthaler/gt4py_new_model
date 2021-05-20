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


def axis(*, length=None, aliases=None):
    def _impl(cls):
        class _axis:
            def __init__(self, index):
                self.index = index

            def __index__(self):
                if length is not None:
                    if self.index >= length:
                        raise IndexError()

                return self.index

            def __str__(self):
                return f"{cls.__name__}({self.index})"

            def __eq__(self, other):
                return type(self) == type(other) and self.index == other.__index__()

        if length is not None:
            setattr(_axis, "__len__", lambda self: length)

        if aliases:
            for i, alias in enumerate(aliases):
                setattr(_axis, alias, _axis(i))

        return _axis

    return _impl


class Dimension:
    def __init__(self, axis, range):
        self.axis = axis
        self.range = range  # range or set of indices or None (=infinite)

    def __eq__(self, other):
        if self.axis == other.axis:
            if self.range is None or other.range is None or self.range == other.range:
                return True
        return False

    def __lt__(self, other):
        if hash(self.axis) < hash(other.axis):
            return True
        elif hash(self.axis) == hash(other.axis):
            if other.range is None:
                return False
            if self.range is None:
                return True
            if self.range < other.range:
                return True
        return False


def make_dimensions(axises, ranges):
    assert len(axises) == len(ranges)
    return tuple(Dimension(a, s) for a, s in zip(axises, ranges))


def dimensions_compatible(first, second):
    lst = list(second)
    for dim in first:
        if dim not in lst:
            return False
        lst.remove(dim)
    return True


def combine_dimensions(first, second):
    # TODO keep order of first?
    res = []
    first = list(first)
    first.sort()
    second = list(second)
    second.sort()
    for f, s in zip(first, second):
        r = s.range if f.range is None else f.range
        res.append(Dimension(f.axis, r))
    return tuple(res)


def order_dimensions(dimensions, ordered_axises):
    res = []
    dims = list(dimensions)
    # oha...
    for axis in ordered_axises:
        for dim in dims:
            if axis == dim.axis:
                res.append(dim)
                dims.remove(dim)
                break
    return tuple(res)


def tupelize(tup):
    if isinstance(tup, tuple):
        return tup
    else:
        return (tup,)


def remove_axises_from_dimensions(to_remove, dimensions):
    res = []
    to_remove = list(to_remove)
    for dim in dimensions:
        if dim.axis in to_remove:
            to_remove.remove(dim.axis)
        else:
            res.append(dim)
    assert len(to_remove) == 0

    return tuple(res)


def remove_indices_of_axises(axises, indices):
    res = indices
    for axis in axises:
        types = tuple(type(i) for i in res)
        i = types.index(axis)
        res = res[:i] + res[i + 1 :]
    return res


def get_index_of_type(axis):
    def fun(indices):
        for ind in indices:
            if isinstance(ind, axis):
                return ind

    return fun


def split_indices(indices, cond_axises):
    true_indices = []
    false_indices = []
    cond_axises = list(cond_axises)
    for i in indices:
        if type(i) in cond_axises:
            true_indices.append(i)
            cond_axises.remove(type(i))
        else:
            false_indices.append(i)
    assert len(cond_axises) == 0
    return tuple(true_indices), tuple(false_indices)
