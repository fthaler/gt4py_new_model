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
    if_,
    sum_reduce,
)
from atlas4py import (
    Topology,
)
import numpy as np
from unstructured.helpers import array_as_field

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


# def nabla_from_sign_stencil(
#     e2v, v2e, pp, S_MXX, S_MYY, vol, node_indices, pole_edges, external_sign
# ):
#     node_indices_of_neighbor_edge = v2e(e2v(node_indices))
#     pole_flag_of_neighbor_edges = v2e(pole_edges)
#     sign_acc = if_(
#         pole_flag_of_neighbor_edges
#         or (
#             broadcast(node_indices, LocationType.Edge)  # ?
#             == node_indices_of_neighbor_edge[0]
#         ),
#         constant_field(1.0, LocationType.Edge),
#         constant_field(-1.0, LocationType.Edge),
#     )

#     # sign_acc = external_sign
#     return make_tuple(
#         compute_pnabla(e2v, v2e, pp, S_MXX, sign_acc, vol),
#         compute_pnabla(e2v, v2e, pp, S_MYY, sign_acc, vol),
#     )


# def test_nabla_from_sign_stencil():
#     mesh, fs_edges, fs_nodes, edges_per_node = make_mesh()

#     pp = make_input_field(mesh, fs_nodes, edges_per_node)
#     S_MXX, S_MYY = make_S(mesh, fs_edges)
#     vol = make_vol(mesh)

#     edge_flags = np.array(mesh.edges.flags())
#     pole_edges = array_to_field(
#         np.array([Topology.check(flag, Topology.POLE) for flag in edge_flags]),
#         LocationType.Edge,
#     )
#     index_field = array_to_field(np.array(range(fs_nodes.size)), LocationType.Vertex)
#     external_sign = make_sign_field(mesh, fs_nodes.size, 7)

#     nodes_domain = list(range(fs_nodes.size))

#     pnabla_MXX = np.zeros((fs_nodes.size))
#     pnabla_MYY = np.zeros((fs_nodes.size))
#     apply_stencil(
#         nabla_from_sign_stencil,
#         [nodes_domain],
#         [
#             make_connectivity_from_atlas(mesh.edges.node_connectivity),
#             make_connectivity_from_atlas(mesh.nodes.edge_connectivity),
#             pp,
#             S_MXX,
#             S_MYY,
#             vol,
#             index_field,
#             pole_edges,
#             external_sign,
#         ],
#         [pnabla_MXX, pnabla_MYY],
#     )

#     assert_close(-3.5455427772566003e-003, min(pnabla_MXX))
#     assert_close(3.5455427772565435e-003, max(pnabla_MXX))
#     assert_close(-3.3540113705465301e-003, min(pnabla_MYY))
#     assert_close(3.3540113705465301e-003, max(pnabla_MYY))


if __name__ == "__main__":
    test_nabla()
    # test_compute_zavgS()
    # test_nabla_from_sign_stencil()
    # print(
    #     "WORK ON accessor of accessor with neihgtable, but simple standalone example to check if derefencing works correctly"
    # )
    # test_acc_of_acc()
