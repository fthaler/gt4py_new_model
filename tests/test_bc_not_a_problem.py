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
import math

from gt4py_new_model import *


@polymorphic_stencil
def left_right_sum(inp, bc_field, index, right_bound):
    if index[()] == 0:
        return bc_field[I - 1] + inp[I + 1]
    elif index[()] == right_bound[()] - 1:
        return inp[I - 1] + bc_field[I + 1]
    else:
        return inp[I - 1] + inp[I + 1]


def test_bc_1level():
    shape = (7, 5, 3)

    @fencil
    def apply(out, inp, bc_field, domain):
        idx = index(shape, "i")
        right_bound = constant(shape[0])
        apply_stencil(left_right_sum, domain, [out], [inp, bc_field, idx, right_bound])

    rng = np.random.default_rng()
    inp = storage(rng.normal(size=shape))
    out = storage(rng.normal(size=shape))
    bc_field = storage(
        rng.normal(size=(shape[0] + 2, shape[1], shape[2])), origin=(1, 0, 0)
    )

    ref = np.empty_like(np.asarray(out))
    ref[-1, :, :] = np.asarray(bc_field)[-1, :, :] + np.asarray(inp)[-2, :, :]
    ref[0, :, :] = np.asarray(inp)[1, :, :] + np.asarray(bc_field)[0, :, :]
    ref[1:-1, :, :] = np.asarray(inp)[:-2, :, :] + np.asarray(inp)[2:, :, :]

    apply(out, inp, bc_field, domain=domain(shape[0], shape[1], shape[2]))
    assert np.all(ref == np.asarray(out)[:, :, :])


def test_bc_2level_fencil():
    shape = (7, 5, 3)

    @fencil
    def apply(out, inp, bc_field, domain):
        idx = index(shape, "i")
        right_bound = constant(shape[0])
        apply_stencil(left_right_sum, domain, [out], [inp, bc_field, idx, right_bound])

    rng = np.random.default_rng()
    inp = storage(rng.normal(size=shape))
    # inp = storage(np.arange(math.prod(shape)).reshape(shape))
    out = storage(rng.normal(size=shape))
    bc_field1 = storage(
        rng.normal(size=(shape[0] + 2, shape[1], shape[2])), origin=(1, 0, 0)
    )
    bc_field2 = storage(
        rng.normal(size=(shape[0] + 2, shape[1], shape[2])), origin=(1, 0, 0)
    )

    ref = np.empty_like(np.asarray(out))
    ref[-1, :, :] = np.asarray(bc_field1)[-1, :, :] + np.asarray(inp)[-2, :, :]
    ref[0, :, :] = np.asarray(inp)[1, :, :] + np.asarray(bc_field1)[0, :, :]
    ref[1:-1, :, :] = np.asarray(inp)[:-2, :, :] + np.asarray(inp)[2:, :, :]
    tmp = ref.copy()
    ref[-1, :, :] = np.asarray(bc_field2)[-1, :, :] + np.asarray(tmp)[-2, :, :]
    ref[0, :, :] = np.asarray(tmp)[1, :, :] + np.asarray(bc_field2)[0, :, :]
    ref[1:-1, :, :] = np.asarray(tmp)[:-2, :, :] + np.asarray(tmp)[2:, :, :]

    tmp_s = storage(rng.normal(size=shape))
    apply(tmp_s, inp, bc_field1, domain=domain(shape[0], shape[1], shape[2]))
    apply(out, tmp_s, bc_field2, domain=domain(shape[0], shape[1], shape[2]))
    assert np.all(ref == np.asarray(out)[:, :, :])


@polymorphic_stencil
def bc_2level_stencil(inp, bc1, bc2, index, right_bound):
    level1 = lift(left_right_sum)(inp, bc1, index, right_bound)
    return left_right_sum(level1, bc2, index, right_bound)


def test_bc_2level_stencil():
    shape = (7, 5, 3)

    @fencil
    def apply(out, inp, bc1, bc2, domain):
        idx = index(shape, "i")
        right_bound = constant(shape[0])
        apply_stencil(
            bc_2level_stencil, domain, [out], [inp, bc1, bc2, idx, right_bound]
        )

    rng = np.random.default_rng()
    inp = storage(rng.normal(size=shape))
    # inp = storage(np.arange(math.prod(shape)).reshape(shape))
    out = storage(rng.normal(size=shape))
    bc_field1 = storage(
        rng.normal(size=(shape[0] + 2, shape[1], shape[2])), origin=(1, 0, 0)
    )
    bc_field2 = storage(
        rng.normal(size=(shape[0] + 2, shape[1], shape[2])), origin=(1, 0, 0)
    )

    ref = np.empty_like(np.asarray(out))
    ref[-1, :, :] = np.asarray(bc_field1)[-1, :, :] + np.asarray(inp)[-2, :, :]
    ref[0, :, :] = np.asarray(inp)[1, :, :] + np.asarray(bc_field1)[0, :, :]
    ref[1:-1, :, :] = np.asarray(inp)[:-2, :, :] + np.asarray(inp)[2:, :, :]
    tmp = ref.copy()
    ref[-1, :, :] = np.asarray(bc_field2)[-1, :, :] + np.asarray(tmp)[-2, :, :]
    ref[0, :, :] = np.asarray(tmp)[1, :, :] + np.asarray(bc_field2)[0, :, :]
    ref[1:-1, :, :] = np.asarray(tmp)[:-2, :, :] + np.asarray(tmp)[2:, :, :]

    apply(out, inp, bc_field1, bc_field2, domain=domain(shape[0], shape[1], shape[2]))
    assert np.all(ref == np.asarray(out)[:, :, :])
