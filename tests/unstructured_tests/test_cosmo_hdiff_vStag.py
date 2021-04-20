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

import enum
from typing import Tuple
import numpy as np

from unstructured.concepts import (
    apply_stencil,
    lift,
    neighborhood,
    stencil,
)
from unstructured.helpers import as_field, simple_connectivity

from .hdiff_reference import hdiff_reference

# (0,0)_u ------------- (0,0)_fx ------------- (1,0)_u
#    |
#    |
#    |
#    |
#    |
#    |
#    |
# (0,0)_fy
#    |
#    |
#    |
#    |
#    |
#    |
#    |
# (0,1)_u ------------- (0,1)_fx ------------- (1,0)_u
#


@enum.unique
class HDiffLocation(enum.IntEnum):
    ULoc = 0
    FluxX = 1
    FluxY = 2


@neighborhood(HDiffLocation.ULoc, HDiffLocation.ULoc)
class U2U:
    shift = (0, 0)


@neighborhood(HDiffLocation.ULoc, HDiffLocation.FluxX)
class U2FX:
    shift = (-0.5, 0)


@neighborhood(HDiffLocation.FluxX, HDiffLocation.ULoc)
class FX2U:
    shift = (0.5, 0)


@neighborhood(HDiffLocation.ULoc, HDiffLocation.FluxY)
class U2FY:
    shift = (0, -0.5)


@neighborhood(HDiffLocation.FluxY, HDiffLocation.ULoc)
class FY2U:
    shift = (0, 0.5)


def make_hdiff_connectivity(neighborhood):
    @simple_connectivity(neighborhood)
    def cartesian_connectivity(*indices):
        class neighs:
            def __getitem__(self, offsets):
                if not isinstance(offsets, tuple):
                    offsets = (offsets,)
                return tuple(
                    map(
                        lambda x: int(x[0] + x[1] + x[2]),
                        zip(indices, offsets, neighborhood.shift),
                    ),
                )

        return neighs()

    return cartesian_connectivity


u2u = U2U()
u2fx = U2FX()
u2fy = U2FY()
fx2u = FX2U()
fy2u = FY2U()


@stencil
def laplacian(inp: U2U):
    return -4 * inp[0, 0] + (inp[-1, 0] + inp[1, 0] + inp[0, -1] + inp[0, 1])


@stencil
def hdiff_flux_x(inp1: FX2U, inp2: Tuple[FX2U, U2U]):
    lap = lift(laplacian)(inp2)
    flux = lap[-0.5, 0] - lap[0.5, 0]  # flux = lap[0, 0] - lap[1, 0]

    return 0 if flux * (inp1[0.5, 0] - inp1[-0.5, 0]) > 0 else flux


@stencil
def hdiff_flux_y(inp1: FY2U, inp2: Tuple[FY2U, U2U]):
    lap = lift(laplacian)(inp2)
    flux = lap[0, -0.5] - lap[0, 0.5]  # flux = lap[0, 0] - lap[0, 1]

    return 0 if flux * (inp1[0, 0.5] - inp1[0, -0.5]) > 0 else flux


@stencil
def hdiff(
    inp_u2u: U2U,
    inp_u2fx2u: Tuple[U2FX, FX2U],
    inp_u2fy2u: Tuple[U2FY, FY2U],
    inp_u2fx2u2u: Tuple[U2FX, FX2U, U2U],
    inp_u2fy2u2u: Tuple[U2FY, FY2U, U2U],
    coeff,
):
    flx = lift(hdiff_flux_x)(inp_u2fx2u, inp_u2fx2u2u)
    fly = lift(hdiff_flux_y)(inp_u2fy2u, inp_u2fy2u2u)
    return inp_u2u[0, 0] - coeff * (
        flx[0.5, 0] - flx[-0.5, 0] + fly[0, 0.5] - fly[0, -0.5]
    )


def test_hdiff(hdiff_reference):
    inp, coeff, out = hdiff_reference
    shape = (inp.shape[0], inp.shape[1])
    inp_s = as_field(inp[:, :, 0], HDiffLocation.ULoc)
    coeff_full_domain = np.zeros(shape)
    coeff_full_domain[2:-2, 2:-2] = coeff[:, :, 0]
    coeff_s = as_field(coeff_full_domain, HDiffLocation.ULoc)
    out_s = np.zeros_like(inp)[:, :, 0]

    inner_domain = [list(range(2, shape[0] - 2)), list(range(2, shape[1] - 2))]

    apply_stencil(
        hdiff,
        inner_domain,
        [
            make_hdiff_connectivity(u2u),
            make_hdiff_connectivity(u2fx),
            make_hdiff_connectivity(u2fy),
            make_hdiff_connectivity(fx2u),
            make_hdiff_connectivity(fy2u),
        ],
        [out_s],
        [inp_s, inp_s, inp_s, inp_s, inp_s, coeff_s],
    )

    assert np.allclose(out[:, :, 0], out_s[2:-2, 2:-2])
