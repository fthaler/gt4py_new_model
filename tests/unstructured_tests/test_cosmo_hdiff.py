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

import math
import numpy as np
import pytest

from unstructured.concepts import (
    LocationType,
    apply_stencil,
    lift,
    neighborhood,
    stencil,
)
from unstructured.helpers import as_1d, as_2d, as_field, simple_connectivity

from .hdiff_reference import hdiff_reference


@neighborhood(LocationType.Vertex, LocationType.Vertex)
class FivePointNeighborHood:
    center = 0
    left = 1
    right = 2
    top = 3
    bottom = 4


fp = FivePointNeighborHood()


def make_fpconn(shape):
    strides = [shape[1], 1]

    @simple_connectivity(fp)
    def fpconn_neighs(index):
        return [
            index,
            index - strides[0],
            index + strides[0],
            index - strides[1],
            index + strides[1],
        ]

    return fpconn_neighs


@stencil((fp,))
def laplacian(inp):
    return -4 * inp[fp.center] + (
        inp[fp.right] + inp[fp.left] + inp[fp.bottom] + inp[fp.top]
    )


@stencil((fp,), (fp, fp))
def hdiff_flux_x(inp1, inp2):
    lap = lift(laplacian)(inp2)
    flux = lap[fp.center] - lap[fp.right]

    return 0 if flux * (inp1[fp.right] - inp1[fp.center]) > 0 else flux


@stencil((fp,), (fp, fp))
def hdiff_flux_y(inp1, inp2):
    lap = lift(laplacian)(inp2)
    flux = lap[fp.center] - lap[fp.bottom]

    return 0 if flux * (inp1[fp.bottom] - inp1[fp.center]) > 0 else flux


@stencil((fp,), (fp, fp), (fp, fp, fp), ())
def hdiff(inp1, inp2, inp3, coeff):
    flx = lift(hdiff_flux_x)(inp2, inp3)
    fly = lift(hdiff_flux_y)(inp2, inp3)
    return inp1[fp.center] - coeff * (
        flx[fp.center] - flx[fp.left] + fly[fp.center] - fly[fp.top]
    )


def test_hdiff(hdiff_reference):
    inp, coeff, out = hdiff_reference
    shape = (inp.shape[0], inp.shape[1])
    inp_s = as_field(as_1d(inp[:, :, 0]), LocationType.Vertex)
    coeff_full_domain = np.zeros(shape)
    coeff_full_domain[2:-2, 2:-2] = coeff[:, :, 0]
    coeff_s = as_field(as_1d(coeff_full_domain), LocationType.Vertex)
    out_s = as_1d(np.zeros_like(inp)[:, :, 0])

    domain = np.arange(math.prod(shape))
    domain_2d = as_2d(domain, shape)
    inner_domain = [as_1d(domain_2d[2:-2, 2:-2]).tolist()]

    apply_stencil(
        hdiff,
        inner_domain,
        [make_fpconn(shape)],
        [out_s],
        [inp_s, inp_s, inp_s, coeff_s],
    )

    assert np.allclose(out[:, :, 0], np.asarray(as_2d(out_s, shape)[2:-2, 2:-2]))
