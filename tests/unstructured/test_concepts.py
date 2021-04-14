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

from unstructured.concepts import (
    ufield,
    LocationType,
    conn_mult,
    connectivity,
    neighborhood,
)


@ufield(LocationType.Vertex)
def dummy_v_field(*indices):
    return 0


@neighborhood(LocationType.Edge, LocationType.Vertex)
class E2VNeighborHood:
    pass


@neighborhood(LocationType.Vertex, LocationType.Edge)
class V2ENeighborHood:
    pass


@connectivity(E2VNeighborHood())
def dummy_e2v_conn(field):
    @ufield(LocationType.Edge)
    def new_field():
        return []

    return new_field


@connectivity(V2ENeighborHood())
def dummy_v2e_conn(field):
    @ufield(LocationType.Vertex)
    def new_field():
        return []

    return new_field


def test_conn_multiply():
    assert dummy_v_field.location == LocationType.Vertex

    assert dummy_e2v_conn.in_location == LocationType.Edge
    assert dummy_e2v_conn.out_location == LocationType.Vertex

    assert dummy_e2v_conn(dummy_v_field).location == LocationType.Edge

    e2v2e = conn_mult(dummy_e2v_conn, dummy_v2e_conn)
    assert e2v2e.in_location == LocationType.Edge
    assert e2v2e.out_location == LocationType.Edge

    e2v2e2v = conn_mult(e2v2e, dummy_e2v_conn)
    assert e2v2e2v.in_location == LocationType.Edge
    assert e2v2e2v.out_location == LocationType.Vertex
