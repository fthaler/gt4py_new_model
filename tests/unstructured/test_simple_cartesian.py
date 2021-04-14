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
from numpy.core.numeric import outer

from unstructured.concepts import (
    LocationType,
    apply_stencil,
    neighborhood,
    stencil,
)
from unstructured.helpers import as_field, simple_connectivity
from unstructured.cartesian import CartesianNeighborHood, cartesian_connectivity

cart = CartesianNeighborHood()


@stencil((cart,))
def laplacian1d(inp):
    return -2 * inp[0] + (inp[-1] + inp[1])


def test_lap1d():
    shape = 10
    inp = np.arange(shape) * np.arange(shape)

    out = np.zeros(shape)
    domain = list(range(1, shape - 1))

    apply_stencil(
        laplacian1d,
        [domain],
        [cartesian_connectivity],
        out,
        [as_field(inp, LocationType.Vertex)],
    )

    ref = np.zeros(shape)
    ref[1:-1] = 2

    assert np.allclose(out, ref)


test_lap1d()


@stencil((cart,))
def laplacian2d(inp):
    return -4 * inp[0, 0] + (inp[-1, 0] + inp[1, 0] + inp[0, -1] + inp[0, 1])


# @stencil((cart,))
# def diff_xr(inp):
#     return inp[1, 0] - inp[0, 0]


# @stencil((cart,))
# def diff_yr(inp):
#     return inp[0, 1] - inp[0, 0]


# @stencil((cart, cart))
# def lap(inp):
#     return diff_xl(lift(diff_xr)(inp)) + diff_yl(lift(diff_yr)(inp))

#     # return (inp[1, 0] - inp[0, 0])[-1, 0] - (inp[1, 0] - inp[0, 0])[0, 0] + ...

#     # return inp[1, 0][-1, 0] - inp[0, 0][-1, 0] + ...


def test_lap():
    shape = (5, 7)
    inp = np.zeros(shape)
    inp[:, :] = np.arange(shape[1]) * np.arange(shape[1])

    out = np.zeros(shape)
    domain = [list(range(1, shape[0] - 1)), list(range(1, shape[1] - 1))]

    apply_stencil(
        laplacian2d,
        domain,
        [cartesian_connectivity],
        out,
        [as_field(inp, LocationType.Vertex)],
    )

    ref = np.zeros(shape)
    ref[1:-1, 1:-1] = 2

    assert np.allclose(out, ref)


test_lap()
