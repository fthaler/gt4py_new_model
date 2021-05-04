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

from abc import ABC, abstractmethod
import itertools
from unstructured.concepts import LocationType, apply_stencil, field_dec
from unstructured.helpers import as_1d, as_2d, array_to_field
import numpy as np
import math


def make_v2v_conn(shape_2d):
    strides = [1, shape_2d[1]]

    class v2v_conn:
        a_connectivity = True

        def __call__(self, field):
            class acc:
                def __getitem__(self, neigh_index):
                    @field_dec(LocationType.Vertex)
                    def v2v_field(field_index):
                        if neigh_index == 0:
                            return field(field_index + strides[0])
                        elif neigh_index == 1:
                            return field(field_index - strides[0])
                        elif neigh_index == 2:
                            return field(field_index + strides[1])
                        elif neigh_index == 3:
                            return field(field_index - strides[1])
                        else:
                            assert False

                    return v2v_field

            return acc()

    return v2v_conn()


def v2v(vv_conn, v_field):
    acc = vv_conn(v_field)
    return acc[0] + acc[1] + acc[2] + acc[3]


def v2v_plus_v(vv_conn, v_field):
    acc = vv_conn(v_field)
    return v_field + acc[0] + acc[1] + acc[2] + acc[3]


def neigh_sum(conn, field):
    acc = conn(field)
    return acc[0] + acc[1]  # + acc[2] + acc[3]


# def v2v(vv_conn, v_field):
#     acc = vv_conn(v2v(vv_conn, v_field))
#     return acc[0] + acc[1]  # + acc[2] + acc[3]


def identity(vv_conn, v_field):
    return v2v(vv_conn, v_field)


def v2v2v(vv_conn, v_field):
    return v2v(vv_conn, v2v(vv_conn, v_field))


def other_fun(e2v_conn, v_field, e_field):
    # return sum(neigh for neigh in e2v_conn(v_field))

    return e_field + e2v_conn(v_field)[0] + e2v_conn(v_field)[1]


def v2e2v(v2e, e2v, v_field, e_field):
    return neigh_sum(v2e, other_fun(e2v, v_field, e_field))


# e2v == vs_from_e


def v2v2v_with_v2v(vv_conn, v_field):
    x = v2v(vv_conn, v_field)
    return v2v(vv_conn, x) + vv_conn(v_field)[0] + vv_conn(v_field)[1]


def test_v2v_plus_v():
    shape = (5, 7)
    v2v_conn = make_v2v_conn(shape)
    inp = np.ones(shape)
    inp1d = as_1d(inp)

    assert v2v_plus_v(v2v_conn, array_to_field(inp1d, LocationType.Vertex))(3) == 5.0


test_v2v_plus_v()


def test_v2v():
    shape = (5, 7)
    # inp1d = np.arange(math.prod(shape))
    # inp = as_2d(inp1d, shape)
    inp = np.ones(shape)
    inp1d = as_1d(inp)
    out1d = np.zeros(math.prod(shape))
    ref = np.zeros(shape)
    ref[1:-1, 1:-1] = np.ones((3, 5)) * 4

    domain = np.arange(math.prod(shape))
    domain_2d = as_2d(domain, shape)
    inner_domain = as_1d(domain_2d[1:-1, 1:-1]).tolist()

    v2v_conn = make_v2v_conn(shape)

    apply_stencil(
        v2v,
        [inner_domain],
        [v2v_conn, array_to_field(inp1d, LocationType.Vertex)],
        [out1d],
    )
    out2d = as_2d(out1d, shape)
    assert np.allclose(out2d, ref)


def test_v2v2v():
    shape = (5, 7)
    # inp = np.random.rand(*shape)
    inp = np.ones(shape)
    out1d = np.zeros(math.prod(shape))
    ref = np.zeros(shape)
    ref[2:-2, 2:-2] = np.ones((1, 3)) * 16

    inp1d = as_1d(inp)

    domain = np.arange(math.prod(shape))
    domain_2d = as_2d(domain, shape)
    inner_domain = as_1d(domain_2d[2:-2, 2:-2]).tolist()

    v2v_conn = make_v2v_conn(shape)
    apply_stencil(
        v2v2v,
        [inner_domain],
        [v2v_conn, array_to_field(inp1d, LocationType.Vertex)],
        [out1d],
    )
    out2d = as_2d(out1d, shape)
    assert np.allclose(out2d, ref)


def test_v2v2v_with_v2v():
    shape = (5, 7)
    # inp = np.random.rand(*shape)
    inp = np.ones(shape)
    out1d = np.zeros(math.prod(shape))
    ref = np.zeros(shape)
    ref[2:-2, 2:-2] = np.ones((1, 3)) * 18

    inp1d = as_1d(inp)

    domain = np.arange(math.prod(shape))
    domain_2d = as_2d(domain, shape)
    inner_domain = as_1d(domain_2d[2:-2, 2:-2]).tolist()

    v2v_conn = make_v2v_conn(shape)
    apply_stencil(
        v2v2v_with_v2v,
        [inner_domain],
        [v2v_conn, array_to_field(inp1d, LocationType.Vertex)],
        [out1d],
    )
    out2d = as_2d(out1d, shape)
    assert np.allclose(out2d, ref)


test_v2v()
test_v2v2v()
test_v2v2v_with_v2v()
