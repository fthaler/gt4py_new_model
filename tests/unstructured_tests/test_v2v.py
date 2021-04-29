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
from unstructured.helpers import as_1d, as_2d, as_field
import numpy as np
import math
import enum


@enum.unique
class LocationType(enum.IntEnum):
    Vertex = 0
    Edge = 1
    Cell = 2


def make_v2v_conn(shape_2d):
    strides = [1, shape_2d[1]]

    class v2v_conn:
        a_connectivity = True

        def __call__(self, field):
            def _field(index):
                return [
                    field(index + strides[0]),
                    field(index - strides[0]),
                    field(index + strides[1]),
                    field(index - strides[1]),
                ]

            return _field

    return v2v_conn()


def is_connectivity(arg):
    return hasattr(arg, "a_connectivity")


def stencil(sten):
    def wrapped_stencil(*connectivities_and_fields):
        def inner(index):
            class new_field:
                def __init__(self, field):
                    self.field = field

                def __call__(self):
                    return self.field(index)

            def wrap_field(_field):
                return new_field(_field)

            def wrap_connectivity(conn):
                def wrapped_conn(field):
                    return new_field(conn(field.field))

                return wrapped_conn

            wrapped_connectivities_and_fields = (
                wrap_connectivity(arg) if is_connectivity(arg) else wrap_field(arg)
                for arg in connectivities_and_fields
            )

            return sten(*wrapped_connectivities_and_fields)

        return inner

    return wrapped_stencil


@stencil
def v2v(vv_conn, v_field):
    acc = vv_conn(v_field)()
    return acc[0] + acc[1] + acc[2] + acc[3]


def v2v_explicit(*connectivities_and_fields):
    def inner(index):
        class new_field:
            def __init__(self, field):
                self.field = field

            def __call__(self):
                return self.field(index)

        def wrap_field(_field):
            return new_field(_field)

        def wrap_connectivity(conn):
            def wrapped_conn(field):
                return new_field(conn(field.field))

            return wrapped_conn

        wrapped_connectivities_and_fields = (
            wrap_connectivity(arg) if is_connectivity(arg) else wrap_field(arg)
            for arg in connectivities_and_fields
        )

        def original(vv_conn, v_field):
            acc = vv_conn(v_field)()
            return acc[0] + acc[1] + acc[2] + acc[3]

        return original(*wrapped_connectivities_and_fields)

    return inner


# @neighborhood(LocationType.Vertex, LocationType.Vertex)
# class V2VNeighborHood:
#     right = 0
#     left = 1
#     top = 2
#     bottom = 3


# def make_v2v_conn(shape_2d):
#     strides = [1, shape_2d[1]]

#     @connectivity(V2VNeighborHood())
#     def v2v_conn(field):
#         @ufield(LocationType.Vertex)
#         def _field(index):
#             return [
#                 field(index + strides[0]),
#                 field(index - strides[0]),
#                 field(index + strides[1]),
#                 field(index - strides[1]),
#             ]

#         return _field

#     return v2v_conn


# vv = V2VNeighborHood()


# @stencil
# def v2v(acc_in: V2VNeighborHood):
#     return acc_in[vv.left] + acc_in[vv.right] + acc_in[vv.top] + acc_in[vv.bottom]


# @stencil
# def v2v2v(acc_in: Tuple[V2VNeighborHood, V2VNeighborHood]):
#     x = lift(v2v)(acc_in)
#     return v2v(x)


# @stencil
# def v2v2v_with_v2v(in2: Tuple[V2VNeighborHood, V2VNeighborHood], in1: V2VNeighborHood):
#     x = lift(v2v)(in2)
#     return v2v(x) + in1[vv.left] + in1[vv.right]


def test_v2v():
    shape = (5, 7)
    # inp = np.random.rand(*shape)
    inp = np.ones(shape)
    out1d = np.zeros(math.prod(shape))
    ref = np.zeros(shape)
    ref[1:-1, 1:-1] = np.ones((3, 5)) * 4

    inp1d = as_1d(inp)

    domain = np.arange(math.prod(shape))
    domain_2d = as_2d(domain, shape)
    inner_domain = as_1d(domain_2d[1:-1, 1:-1]).tolist()

    v2v_conn = make_v2v_conn(shape)

    print(v2v_explicit(v2v_conn, as_field(inp1d, LocationType.Vertex))(1))
    print(v2v(v2v_conn, as_field(inp1d, LocationType.Vertex))(1))

    # apply_stencil(
    #     v2v, [inner_domain], [v2v_conn], [out1d], [as_field(inp1d, LocationType.Vertex)]
    # )
    # out2d = as_2d(out1d, shape)
    # assert np.allclose(out2d, ref)


test_v2v()

# def test_v2v2v():
#     shape = (5, 7)
#     # inp = np.random.rand(*shape)
#     inp = np.ones(shape)
#     out1d = np.zeros(math.prod(shape))
#     ref = np.zeros(shape)
#     ref[2:-2, 2:-2] = np.ones((1, 3)) * 16

#     inp1d = as_1d(inp)

#     domain = np.arange(math.prod(shape))
#     domain_2d = as_2d(domain, shape)
#     inner_domain = as_1d(domain_2d[2:-2, 2:-2]).tolist()

#     v2v_conn = make_v2v_conn(shape)
#     apply_stencil(
#         v2v2v,
#         [inner_domain],
#         [v2v_conn],
#         [out1d],
#         [as_field(inp1d, LocationType.Vertex)],
#     )
#     out2d = as_2d(out1d, shape)
#     assert np.allclose(out2d, ref)


# def test_v2v2v_with_v2v():
#     shape = (5, 7)
#     # inp = np.random.rand(*shape)
#     inp = np.ones(shape)
#     out1d = np.zeros(math.prod(shape))
#     ref = np.zeros(shape)
#     ref[2:-2, 2:-2] = np.ones((1, 3)) * 18

#     inp1d = as_1d(inp)

#     domain = np.arange(math.prod(shape))
#     domain_2d = as_2d(domain, shape)
#     inner_domain = as_1d(domain_2d[2:-2, 2:-2]).tolist()

#     v2v_conn = make_v2v_conn(shape)
#     apply_stencil(
#         v2v2v_with_v2v,
#         [inner_domain],
#         [v2v_conn],
#         [out1d],
#         [as_field(inp1d, LocationType.Vertex), as_field(inp1d, LocationType.Vertex)],
#     )
#     out2d = as_2d(out1d, shape)
#     assert np.allclose(out2d, ref)


# test_v2v()
# test_v2v2v()
# test_v2v2v_with_v2v()
