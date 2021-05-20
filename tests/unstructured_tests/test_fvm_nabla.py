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
from typing import Generic, TypeVar
from unstructured.concepts import (
    broadcast,
    if_,
    sum_reduce,
)
from atlas4py import (
    Topology,
)
import numpy as np
from unstructured.helpers import (
    array_as_field,
    constant_field,
    index_field,
    materialize,
)

from unstructured.utils import axis

from .fvm_nabla_setup import (
    assert_close,
    nabla_setup,
)

from unstructured.atlas_utils import make_sparse_index_field_from_atlas_connectivity


@axis()
class Vertex:
    pass


@axis()
class Edge:
    pass


@axis()
class V2E:
    pass


@axis()
class E2V:
    pass


def stencil(fun):
    def _fun(*args):
        return materialize(fun(*args))

    return _fun


class AccMeta(type):
    def __getitem__(cls, _):
        pass


class Acc(metaclass=AccMeta):
    pass


def lift(sten):
    return sten


@stencil
def compute_zavgS(
    pp_neighs,
    S_M
    # pp_neighs: Acc[E2V],
    # S_M: float,
):  # type annotations are wrong, but just to illustrate that there is literaly no difference between global and local view
    zavg = 0.5 * (pp_neighs[E2V(0)] + pp_neighs[E2V(1)])
    # zavg = 0.5 * sum_reduce(E2V)(pp_neighs)
    return S_M * zavg
    # return materialize(S_M * zavg)


@stencil
def compute_pnabla(e2v, v2e, pp, S_M, sign, vol):
    zavgS = compute_zavgS(pp[e2v], S_M)[v2e]
    # zavgS = lift(compute_zavgS(pp[e2v], S_M))[v2e]
    # pnabla_M = sum_reduce(V2E)(zavgS * sign)
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


def test_compute_zavgS():
    setup = nabla_setup()

    pp = array_as_field(Vertex)(setup.input_field)
    S_MXX, S_MYY = tuple(map(array_as_field(Edge), setup.S_fields))

    zavgS = np.zeros((setup.edges_size))

    e2v = make_sparse_index_field_from_atlas_connectivity(
        setup.edges2node_connectivity, Edge, E2V, Vertex
    )

    zavgS[:] = compute_zavgS(pp[e2v], S_MXX)

    assert_close(-199755464.25741270, min(zavgS))
    assert_close(388241977.58389181, max(zavgS))

    zavgS[:] = compute_zavgS(pp[e2v], S_MYY)
    assert_close(-1000788897.3202186, min(zavgS))
    assert_close(1000788897.3202186, max(zavgS))


def test_nabla():
    setup = nabla_setup()

    sign_acc = array_as_field(Vertex, V2E)(setup.sign_field)
    pp = array_as_field(Vertex)(setup.input_field)
    S_MXX, S_MYY = tuple(map(array_as_field(Edge), setup.S_fields))
    vol = array_as_field(Vertex)(setup.vol_field)

    pnabla_MXX = np.zeros((setup.nodes_size))
    pnabla_MYY = np.zeros((setup.nodes_size))

    print(f"nodes: {setup.nodes_size}")
    print(f"edges: {setup.edges_size}")

    e2v = make_sparse_index_field_from_atlas_connectivity(
        setup.edges2node_connectivity, Edge, E2V, Vertex
    )
    v2e = make_sparse_index_field_from_atlas_connectivity(
        setup.nodes2edge_connectivity, Vertex, V2E, Edge
    )

    pnabla_MXX[:], pnabla_MYY[:] = nabla(e2v, v2e, pp, S_MXX, S_MYY, sign_acc, vol)

    assert_close(-3.5455427772566003e-003, min(pnabla_MXX))
    assert_close(3.5455427772565435e-003, max(pnabla_MXX))
    assert_close(-3.3540113705465301e-003, min(pnabla_MYY))
    assert_close(3.3540113705465301e-003, max(pnabla_MYY))


@stencil
def sign(e2v, v2e, node_indices, is_pole_edge):
    node_indices_of_neighbor_edge = node_indices[e2v[v2e]]
    pole_flag_of_neighbor_edges = is_pole_edge[v2e]
    sign_field = if_(
        pole_flag_of_neighbor_edges
        | (broadcast(V2E)(node_indices) == node_indices_of_neighbor_edge[E2V(0)]),
        constant_field(Vertex, V2E)(1.0),
        constant_field(Vertex, V2E)(-1.0),
    )
    return sign_field


@stencil
def compute_pnabla_sign(e2v, v2e, pp, S_M, node_indices, is_pole_edge, vol):
    zavgS = compute_zavgS(pp[e2v], S_M)[v2e]
    pnabla_M = sum_reduce(V2E)(zavgS * sign(e2v, v2e, node_indices, is_pole_edge))

    return pnabla_M / vol


def nabla_sign(
    e2v,
    v2e,
    pp,
    S_MXX,
    S_MYY,
    node_indices,
    is_pole_edge,
    vol,
):
    return (
        compute_pnabla_sign(e2v, v2e, pp, S_MXX, node_indices, is_pole_edge, vol),
        compute_pnabla_sign(e2v, v2e, pp, S_MYY, node_indices, is_pole_edge, vol),
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

    node_index_field = index_field(Vertex, range(setup.nodes_size))

    e2v = make_sparse_index_field_from_atlas_connectivity(
        setup.edges2node_connectivity, Edge, E2V, Vertex
    )
    v2e = make_sparse_index_field_from_atlas_connectivity(
        setup.nodes2edge_connectivity, Vertex, V2E, Edge
    )

    pnabla_MXX = np.zeros((setup.nodes_size))
    pnabla_MYY = np.zeros((setup.nodes_size))

    print(f"nodes: {setup.nodes_size}")
    print(f"edges: {setup.edges_size}")

    pnabla_MXX[:], pnabla_MYY[:] = nabla_sign(
        e2v, v2e, pp, S_MXX, S_MYY, node_index_field, is_pole_edge, vol
    )

    assert_close(-3.5455427772566003e-003, min(pnabla_MXX))
    assert_close(3.5455427772565435e-003, max(pnabla_MXX))
    assert_close(-3.3540113705465301e-003, min(pnabla_MYY))
    assert_close(3.3540113705465301e-003, max(pnabla_MYY))


@stencil
def compute_pnabla_on_nodes(v2e, v2e_e2v_pp, S_M, sign, vol):
    zavgS = S_M[v2e] * 0.5 * (v2e_e2v_pp[E2V(0)] + v2e_e2v_pp[E2V(1)])
    pnabla_M = sum_reduce(V2E)(zavgS * sign)

    return pnabla_M / vol


def nabla_on_nodes(
    v2e2v,
    v2e,
    pp,
    S_MXX,
    S_MYY,
    sign,
    vol,
):
    v2e_e2v_pp = materialize(pp[v2e2v])
    return (
        compute_pnabla_on_nodes(v2e, v2e_e2v_pp, S_MXX, sign, vol),
        compute_pnabla_on_nodes(v2e, v2e_e2v_pp, S_MYY, sign, vol),
    )


def test_nabla_on_nodes():
    setup = nabla_setup()

    sign_acc = array_as_field(Vertex, V2E)(setup.sign_field)
    pp = array_as_field(Vertex)(setup.input_field)
    S_MXX, S_MYY = tuple(map(array_as_field(Edge), setup.S_fields))
    vol = array_as_field(Vertex)(setup.vol_field)

    e2v = make_sparse_index_field_from_atlas_connectivity(
        setup.edges2node_connectivity, Edge, E2V, Vertex
    )
    v2e = make_sparse_index_field_from_atlas_connectivity(
        setup.nodes2edge_connectivity, Vertex, V2E, Edge
    )
    v2e2v = e2v[
        v2e
    ]  # TODO materialize is broken because I don't preserver optional if materialized as np array

    pnabla_MXX = np.zeros((setup.nodes_size))
    pnabla_MYY = np.zeros((setup.nodes_size))

    print(f"nodes: {setup.nodes_size}")
    print(f"edges: {setup.edges_size}")

    pnabla_MXX[:], pnabla_MYY[:] = nabla_on_nodes(
        v2e2v, v2e, pp, S_MXX, S_MYY, sign_acc, vol
    )

    assert_close(-3.5455427772566003e-003, min(pnabla_MXX))
    assert_close(3.5455427772565435e-003, max(pnabla_MXX))
    assert_close(-3.3540113705465301e-003, min(pnabla_MYY))
    assert_close(3.3540113705465301e-003, max(pnabla_MYY))
