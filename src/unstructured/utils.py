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


def _tupelize(tup):
    if isinstance(tup, tuple):
        return tup
    else:
        return (tup,)


def remove_axis(axis, axises):
    i = axises.index(axis)
    return axises[:i] + axises[i + 1 :]


def remove_axises_from_axises(to_remove, axises):
    res = axises
    for axis in to_remove:
        res = remove_axis(axis, res)
    return res


def remove_indices_of_axises(axises, indices):
    res = indices
    for axis in axises:
        types = tuple(type(i) for i in res)
        i = types.index(axis)
        res = res[:i] + res[i + 1 :]
    return res


def get_index_of_type(loc):
    def fun(indices):
        for ind in indices:
            if isinstance(ind, loc):
                return ind

    return fun


def print_axises(axises):
    print([str(axis(0)) for axis in axises])
