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

from gt4py_new_model.storage import storage, index


def test_storage():
    rng = np.random.default_rng()
    array = rng.normal(size=(2, 3))
    s = storage(array)
    assert np.all(np.asarray(s) == array)


def test_index():
    i, j = np.indices((2, 3))
    i_s = index((2, 3), "i")
    j_s = index((2, 3), "j")
    assert np.all(np.asarray(i_s) == i)
    assert np.all(np.asarray(j_s) == j)
