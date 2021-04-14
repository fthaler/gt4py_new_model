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

import numpy as np

from gt4py_new_model.accessors import array_column_accessor
from gt4py_new_model.dimensions import K
from gt4py_new_model.builtins import forward, backward


def test_scan():
    def acc(column):
        return array_column_accessor(
            array=np.array(column),
            dimensions=("k",),
            focus=(0,),
            column="k",
            size=len(column),
        )

    @forward
    def identity(_, x):
        return x[K]

    @backward
    def reverse(_, x):
        return x[K]

    assert np.all(identity(acc([1, 2])) == np.array([1, 2]))
    assert np.all(reverse(acc([1, 2])) == np.array([1, 2]))

    @forward(init=1)
    def colsum(state, x):
        return state + x[K]

    @backward(init=1)
    def rcolsum(state, x):
        return state + x[K]

    assert np.all(colsum(acc([1, 2])) == np.array([2, 4]))
    assert np.all(rcolsum(acc([1, 2])) == np.array([4, 3]))

    @forward(init=(1, 1))
    def tup(state, x):
        return state[0] + x[K], state[1] - x[K]

    a, b = tup(acc([1, 2]))
    assert np.all(a == np.array([2, 4]))
    assert np.all(b == np.array([0, -2]))
