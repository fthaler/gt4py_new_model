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

from unstructured.concepts import (
    LocationType,
    apply_stencil,
    connectivity,
    lift,
    stencil,
    ufield,
)
from unstructured.helpers import as_field
from unstructured.cartesian import CartesianNeighborHood

from .hdiff_reference import hdiff_reference

cart = CartesianNeighborHood()


def cartesian_accessor(field, *indices):
    class _cartesian_accessor:
        def __call__(self):
            return field(*indices)

        def __getitem__(self, neighindices):
            if not isinstance(neighindices, tuple):
                neighindices = (neighindices,)
            return cartesian_accessor(
                field,
                *tuple(
                    map(lambda x: x[0] + x[1], zip(indices, neighindices)),
                )
            )

    return _cartesian_accessor()


@connectivity(cart)
def all_rank_cartesian_connectivity(field):
    @ufield(cart.in_location)
    def _field(*index):

        return cartesian_accessor(field, *index)

    return _field


@stencil
def laplacian(inp: CartesianNeighborHood):
    return -4 * inp[0, 0]() + (inp[-1, 0]() + inp[1, 0]() + inp[0, -1]() + inp[0, 1]())


@stencil
def hdiff_flux_x(inp: CartesianNeighborHood):
    lap = lift(laplacian)(inp)
    flux = lap[0, 0] - lap[1, 0]

    return 0 if flux * (inp[1, 0]() - inp[0, 0]()) > 0 else flux


@stencil
def hdiff_flux_y(inp: CartesianNeighborHood):
    lap = lift(laplacian)(inp)
    flux = lap[0, 0] - lap[0, 1]

    return 0 if flux * (inp[0, 1]() - inp[0, 0]()) > 0 else flux


@stencil
def hdiff(inp: CartesianNeighborHood, coeff):
    flx = lift(hdiff_flux_x)(inp)
    fly = lift(hdiff_flux_y)(inp)
    return inp[0, 0]() - coeff * (flx[0, 0] - flx[-1, 0] + fly[0, 0] - fly[0, -1])


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
        [all_rank_cartesian_connectivity],
        [out_s],
        [inp_s, coeff_s],
    )

    assert np.allclose(out[:, :, 0], out_s[2:-2, 2:-2])
