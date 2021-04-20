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

from typing import Tuple
import numpy as np

from unstructured.concepts import (
    LocationType,
    apply_stencil,
    lift,
    stencil,
)
from unstructured.helpers import as_field
from unstructured.cartesian import CartesianNeighborHood, cartesian_connectivity

from .hdiff_reference import hdiff_reference

cart = CartesianNeighborHood()


@stencil
def laplacian(inp: CartesianNeighborHood):
    return -4 * inp[0, 0] + (inp[-1, 0] + inp[1, 0] + inp[0, -1] + inp[0, 1])


@stencil
def hdiff_flux_x(
    inp1: CartesianNeighborHood,
    inp2: Tuple[CartesianNeighborHood, CartesianNeighborHood],
):
    lap = lift(laplacian)(inp2)
    flux = lap[0, 0] - lap[1, 0]

    return 0 if flux * (inp1[1, 0] - inp1[0, 0]) > 0 else flux


@stencil
def hdiff_flux_y(
    inp1: CartesianNeighborHood,
    inp2: Tuple[CartesianNeighborHood, CartesianNeighborHood],
):
    lap = lift(laplacian)(inp2)
    flux = lap[0, 0] - lap[0, 1]

    return 0 if flux * (inp1[0, 1] - inp1[0, 0]) > 0 else flux


@stencil
def hdiff(
    inp1: CartesianNeighborHood,
    inp2: Tuple[CartesianNeighborHood, CartesianNeighborHood],
    inp3: Tuple[CartesianNeighborHood, CartesianNeighborHood, CartesianNeighborHood],
    coeff,
):
    flx = lift(hdiff_flux_x)(inp2, inp3)
    fly = lift(hdiff_flux_y)(inp2, inp3)
    return inp1[0, 0] - coeff * (flx[0, 0] - flx[-1, 0] + fly[0, 0] - fly[0, -1])


def test_hdiff(hdiff_reference):
    inp, coeff, out = hdiff_reference
    shape = (inp.shape[0], inp.shape[1])
    inp_s = as_field(inp[:, :, 0], LocationType.Vertex)
    coeff_full_domain = np.zeros(shape)
    coeff_full_domain[2:-2, 2:-2] = coeff[:, :, 0]
    coeff_s = as_field(coeff_full_domain, LocationType.Vertex)
    out_s = np.zeros_like(inp)[:, :, 0]

    inner_domain = [list(range(2, shape[0] - 2)), list(range(2, shape[1] - 2))]

    apply_stencil(
        hdiff,
        inner_domain,
        [cartesian_connectivity],
        [out_s],
        [inp_s, inp_s, inp_s, coeff_s],
    )

    assert np.allclose(out[:, :, 0], out_s[2:-2, 2:-2])
