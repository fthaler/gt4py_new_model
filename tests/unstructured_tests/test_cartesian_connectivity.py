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
    connectivity,
    neighborhood,
    stencil,
    ufield,
)
from unstructured.helpers import as_field, simple_connectivity
from unstructured.cartesian import CartesianNeighborHood

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
def cartesian_connectivity2(field):
    @ufield(cart.in_location)
    def _field(*index):

        return cartesian_accessor(field, *index)

    return _field


def test_cartesian_connectivity():
    inp = np.arange(10 * 10).reshape(10, 10)
    print(inp)

    inp_s = as_field(inp, LocationType.Vertex)

    assert cartesian_accessor(inp_s, 1, 1)[1, 1][2, 2]() == 44
    acc_field = cartesian_connectivity2(inp_s)

    assert inp_s(1, 1) == 11
    assert acc_field(1, 1)[1, 1]() == 22
    assert acc_field(1, 1)[1, 1][2, 2]() == 44


test_cartesian_connectivity()
