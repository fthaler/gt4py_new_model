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
    apply_stencil,
    broadcast,
    if_,
    sum_reduce,
)
from atlas4py import (
    Topology,
)
import numpy as np
from unstructured.helpers import array_as_field, constant_field, index_field

from unstructured.utils import axis

from .fvm_nabla_setup import (
    assert_close,
    nabla_setup,
)

from unstructured.atlas_utils import make_sparse_index_field_from_atlas_connectivity


@axis(length=None)
class Vertex:
    pass


@axis(length=None)
class Edge:
    pass


@axis(length=7)
class V2E:
    pass


@axis(length=2)
class E2V:
    pass


def compute_zavgS(e2v, pp, S_M):
    pp_neighs = e2v(pp)
    zavg = 0.5 * (pp_neighs[E2V(0)] + pp_neighs[E2V(1)])
    # zavg = 0.5 * sum_reduce(E2V)(pp_neighs)
    return S_M * zavg


def compute_pnabla(e2v, v2e, pp, S_M, sign, vol):
    zavgS = v2e(compute_zavgS(e2v, pp, S_M))
    pnabla_M = sum_reduce(V2E)(zavgS * sign)

    return pnabla_M / vol


def nabla(
    e2v,
    v2e,
    pp,
    S_MXX,
    S_MYY,
    sign,
    vol,
):
    return compute_pnabla(e2v, v2e, pp, S_MXX, sign, vol), compute_pnabla(
        e2v, v2e, pp, S_MYY, sign, vol
    )


def make_connectivity(index_field):
    return lambda field: field[index_field]


def test_compute_zavgS():
    setup = nabla_setup()

    pp = array_as_field(Vertex)(setup.input_field)
    S_MXX, S_MYY = tuple(map(array_as_field(Edge), setup.S_fields))

    edge_domain = list(range(setup.edges_size))

    zavgS = np.zeros((setup.edges_size))

    e2v_conn = make_connectivity(
        make_sparse_index_field_from_atlas_connectivity(
            setup.edges2node_connectivity, Edge, E2V, Vertex
        )
    )

    apply_stencil(
        compute_zavgS,
        [(edge_domain, Edge)],
        [e2v_conn, pp, S_MXX],
        [zavgS],
    )
    assert_close(-199755464.25741270, min(zavgS))
    assert_close(388241977.58389181, max(zavgS))

    apply_stencil(
        compute_zavgS,
        [(edge_domain, Edge)],
        [e2v_conn, pp, S_MYY],
        [zavgS],
    )
    assert_close(-1000788897.3202186, min(zavgS))
    assert_close(1000788897.3202186, max(zavgS))


def test_nabla():
    setup = nabla_setup()

    sign_acc = array_as_field(Vertex, V2E)(setup.sign_field)
    pp = array_as_field(Vertex)(setup.input_field)
    S_MXX, S_MYY = tuple(map(array_as_field(Edge), setup.S_fields))
    vol = array_as_field(Vertex)(setup.vol_field)

    nodes_domain = list(range(setup.nodes_size))

    pnabla_MXX = np.zeros((setup.nodes_size))
    pnabla_MYY = np.zeros((setup.nodes_size))

    print(f"nodes: {setup.nodes_size}")
    print(f"edges: {setup.edges_size}")

    e2v_conn = make_connectivity(
        make_sparse_index_field_from_atlas_connectivity(
            setup.edges2node_connectivity, Edge, E2V, Vertex
        )
    )
    v2e_conn = make_connectivity(
        make_sparse_index_field_from_atlas_connectivity(
            setup.nodes2edge_connectivity, Vertex, V2E, Edge
        )
    )

    apply_stencil(
        nabla,
        [(nodes_domain, Vertex)],
        [
            e2v_conn,
            v2e_conn,
            pp,
            S_MXX,
            S_MYY,
            sign_acc,
            vol,
        ],
        [pnabla_MXX, pnabla_MYY],
    )

    assert_close(-3.5455427772566003e-003, min(pnabla_MXX))
    assert_close(3.5455427772565435e-003, max(pnabla_MXX))
    assert_close(-3.3540113705465301e-003, min(pnabla_MYY))
    assert_close(3.3540113705465301e-003, max(pnabla_MYY))


setup = nabla_setup()

e2v_field = make_sparse_index_field_from_atlas_connectivity(
    setup.edges2node_connectivity, Edge, E2V, Vertex
)
v2e_field = make_sparse_index_field_from_atlas_connectivity(
    setup.nodes2edge_connectivity, Vertex, V2E, Edge
)

e2v_conn = make_connectivity(e2v_field)
v2e_conn = make_connectivity(v2e_field)


def compute_zavgS_glob(pp, S_M):
    pp_neighs = pp[e2v_field]
    # pp_neighs = e2v_conn(pp)
    zavg = 0.5 * (pp_neighs[E2V(0)] + pp_neighs[E2V(1)])
    return S_M * zavg


def compute_pnabla_glob(pp, S_M, sign, vol):
    zavgS = compute_zavgS_glob(pp, S_M)[v2e_field]
    # zavgS = v2e_conn(compute_zavgS_glob(pp, S_M))
    pnabla_M = sum_reduce(V2E)(zavgS * sign)

    return pnabla_M / vol


def nabla_glob(
    pp,
    S_MXX,
    S_MYY,
    sign,
    vol,
):
    return compute_pnabla_glob(pp, S_MXX, sign, vol), compute_pnabla_glob(
        pp, S_MYY, sign, vol
    )


def test_nabla_global_index_fields():
    setup = nabla_setup()

    sign_acc = array_as_field(Vertex, V2E)(setup.sign_field)
    pp = array_as_field(Vertex)(setup.input_field)
    S_MXX, S_MYY = tuple(map(array_as_field(Edge), setup.S_fields))
    vol = array_as_field(Vertex)(setup.vol_field)

    nodes_domain = list(range(setup.nodes_size))

    pnabla_MXX = np.zeros((setup.nodes_size))
    pnabla_MYY = np.zeros((setup.nodes_size))

    print(f"nodes: {setup.nodes_size}")
    print(f"edges: {setup.edges_size}")

    apply_stencil(
        nabla_glob,
        [(nodes_domain, Vertex)],
        [
            pp,
            S_MXX,
            S_MYY,
            sign_acc,
            vol,
        ],
        [pnabla_MXX, pnabla_MYY],
    )

    assert_close(-3.5455427772566003e-003, min(pnabla_MXX))
    assert_close(3.5455427772565435e-003, max(pnabla_MXX))
    assert_close(-3.3540113705465301e-003, min(pnabla_MYY))
    assert_close(3.3540113705465301e-003, max(pnabla_MYY))


def sign(node_indices, is_pole_edge):
    node_indices_of_neighbor_edge = node_indices[e2v_field[v2e_field]]
    pole_flag_of_neighbor_edges = is_pole_edge[v2e_field]
    sign_field = if_(
        pole_flag_of_neighbor_edges
        | (broadcast(V2E)(node_indices) == node_indices_of_neighbor_edge[E2V(0)]),
        constant_field(Vertex, V2E)(1.0),
        constant_field(Vertex, V2E)(-1.0),
    )
    return sign_field


def compute_zavgS_sign(pp, S_M):
    pp_neighs = pp[e2v_field]
    zavg = 0.5 * (pp_neighs[E2V(0)] + pp_neighs[E2V(1)])
    return S_M * zavg


def compute_pnabla_sign(pp, S_M, node_indices, is_pole_edge, vol):
    zavgS = compute_zavgS_sign(pp, S_M)[v2e_field]
    pnabla_M = sum_reduce(V2E)(zavgS * sign(node_indices, is_pole_edge))

    return pnabla_M / vol


def nabla_sign(
    pp,
    S_MXX,
    S_MYY,
    node_indices,
    is_pole_edge,
    vol,
):
    return (
        compute_pnabla_sign(pp, S_MXX, node_indices, is_pole_edge, vol),
        compute_pnabla_sign(pp, S_MYY, node_indices, is_pole_edge, vol),
    )


def test_nabla_from_sign_stencil():
    setup = nabla_setup()

    pp = array_as_field(Vertex)(setup.input_field)
    S_MXX, S_MYY = tuple(map(array_as_field(Edge), setup.S_fields))
    vol = array_as_field(Vertex)(setup.vol_field)

    edge_flags = np.array(setup.mesh.edges.flags())
    is_pole_edge = array_as_field(Edge)(
        np.array([Topology.check(flag, Topology.POLE) for flag in edge_flags])
    )

    node_index_field = index_field(Vertex)

    nodes_domain = list(range(setup.nodes_size))

    pnabla_MXX = np.zeros((setup.nodes_size))
    pnabla_MYY = np.zeros((setup.nodes_size))

    print(f"nodes: {setup.nodes_size}")
    print(f"edges: {setup.edges_size}")

    apply_stencil(
        nabla_sign,
        [(nodes_domain, Vertex)],
        [
            pp,
            S_MXX,
            S_MYY,
            node_index_field,
            is_pole_edge,
            vol,
        ],
        [pnabla_MXX, pnabla_MYY],
    )

    assert_close(-3.5455427772566003e-003, min(pnabla_MXX))
    assert_close(3.5455427772565435e-003, max(pnabla_MXX))
    assert_close(-3.3540113705465301e-003, min(pnabla_MYY))
    assert_close(3.3540113705465301e-003, max(pnabla_MYY))
