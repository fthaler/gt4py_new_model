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
from unstructured.concepts import (
    LocationType,
    accessor,
    apply_stencil,
    lift,
    neighborhood,
    stencil,
    ufield,
)
from unstructured.helpers import as_field, simple_connectivity
from atlas4py import (
    StructuredGrid,
    Topology,
    Config,
    StructuredMeshGenerator,
    functionspace,
    build_edges,
    build_node_to_edge_connectivity,
    build_median_dual_mesh,
    IrregularConnectivity,
)
import numpy as np
import math


@neighborhood(LocationType.Edge, LocationType.Vertex)
class Edge2Vertex:
    pass


@neighborhood(LocationType.Vertex, LocationType.Edge)
class Vertex2Edge:
    pass


e2v = Edge2Vertex()
v2e = Vertex2Edge()


def make_connectivity_from_atlas(neightbl, neighborhood):
    @simple_connectivity(neighborhood)
    def neighs(index):
        class impl:
            def __getitem__(self, neighindex):
                if isinstance(neightbl, IrregularConnectivity):
                    # print(f"neighindex {neighindex}, cols {neightbl.cols(index)}")
                    if neighindex < neightbl.cols(index):
                        return neightbl[index, neighindex]
                    else:
                        return None
                else:
                    assert isinstance(neighborhood, Edge2Vertex)
                    if neighindex < 2:
                        return neightbl[index, neighindex]
                    else:
                        assert False

        return impl()

    return neighs


def atlas2d_to_field(atlas_field, location_type):
    # make column field a pure horizontal field
    return as_field(np.array(atlas_field, copy=False)[:, 0], location_type)


def sparsefield_to_accessor_field(array, neighborhood):
    @ufield(neighborhood.in_location)
    def _field(index):
        @accessor(neighborhood)
        def acc(neigh):
            return array[index, neigh]

        return acc

    return _field


def make_mesh():
    edges_per_node = 7  # TODO

    grid = StructuredGrid("O32")
    config = Config()
    config["triangulate"] = True
    config["angle"] = 20.0

    mesh = StructuredMeshGenerator(config).generate(grid)

    fs_edges = functionspace.EdgeColumns(mesh, halo=1)
    fs_nodes = functionspace.NodeColumns(mesh, halo=1)

    build_edges(mesh)
    build_node_to_edge_connectivity(mesh)
    build_median_dual_mesh(mesh)

    return mesh, fs_edges, fs_nodes, edges_per_node


@stencil
def sign_stencil(node_id, pole_edge, nodes_indices: Edge2Vertex):
    if pole_edge or node_id == nodes_indices[0]:
        return 1.0
    else:
        return -1.0


def make_sign_field(mesh, nodes_size, edges_per_node):
    node2edge_sign = np.zeros((nodes_size, edges_per_node))
    edge_flags = np.array(mesh.edges.flags())

    def is_pole_edge(e):
        return Topology.check(edge_flags[e], Topology.POLE)

    for jnode in range(0, nodes_size):
        node_edge_con = mesh.nodes.edge_connectivity
        edge_node_con = mesh.edges.node_connectivity
        for jedge in range(0, node_edge_con.cols(jnode)):
            iedge = node_edge_con[jnode, jedge]
            ip1 = edge_node_con[iedge, 0]
            if jnode == ip1:
                node2edge_sign[jnode, jedge] = 1.0
            else:
                node2edge_sign[jnode, jedge] = -1.0
                if is_pole_edge(iedge):
                    node2edge_sign[jnode, jedge] = 1.0
    return sparsefield_to_accessor_field(node2edge_sign, v2e)


def assert_close(expected, actual):
    assert math.isclose(expected, actual), "expected={}, actual={}".format(
        expected, actual
    )


def test_sign_field():
    @stencil
    def validate_sign(
        node_index,
        pole_edges: Vertex2Edge,
        nodes_indices: Tuple[Vertex2Edge, Edge2Vertex],
        external_sign: Vertex2Edge,
    ):
        sign_acc = lift(sign_stencil)(node_index, pole_edges, nodes_indices)
        for i in range(7):
            if sign_acc[i] and external_sign[i]:
                assert sign_acc[i] == external_sign[i]

    mesh, fs_edges, fs_nodes, edges_per_node = make_mesh()
    sign_acc = make_sign_field(mesh, fs_nodes.size, edges_per_node)  # acc of rank 1

    edge_flags = np.array(mesh.edges.flags())
    pole_edges = as_field(
        np.array([Topology.check(flag, Topology.POLE) for flag in edge_flags]),
        LocationType.Edge,
    )
    index_field = as_field(np.array(range(fs_nodes.size)), LocationType.Vertex)

    nodes_domain = list(range(fs_nodes.size))

    out = np.zeros((fs_nodes.size,))

    apply_stencil(
        validate_sign,
        [nodes_domain],
        [
            make_connectivity_from_atlas(mesh.edges.node_connectivity, e2v),
            make_connectivity_from_atlas(mesh.nodes.edge_connectivity, v2e),
        ],
        [out],
        [index_field, pole_edges, index_field, sign_acc],
    )


def make_S(mesh, fs_edges):
    S = np.array(mesh.edges.field("dual_normals"), copy=False)
    S_MXX = np.zeros((fs_edges.size))
    S_MYY = np.zeros((fs_edges.size))

    MXX = 0
    MYY = 1

    rpi = 2.0 * math.asin(1.0)
    radius = 6371.22e03
    deg2rad = 2.0 * rpi / 360.0

    for i in range(0, fs_edges.size):
        S_MXX[i] = S[i, MXX] * radius * deg2rad
        S_MYY[i] = S[i, MYY] * radius * deg2rad

    assert math.isclose(min(S_MXX), -103437.60479272791)
    assert math.isclose(max(S_MXX), 340115.33913622628)
    assert math.isclose(min(S_MYY), -2001577.7946404363)
    assert math.isclose(max(S_MYY), 2001577.7946404363)

    return as_field(S_MXX, LocationType.Edge), as_field(S_MYY, LocationType.Edge)


def make_vol(mesh):
    rpi = 2.0 * math.asin(1.0)
    radius = 6371.22e03
    deg2rad = 2.0 * rpi / 360.0
    vol_atlas = np.array(mesh.nodes.field("dual_volumes"), copy=False)
    # dual_volumes 4.6510228700066421    68.891611253882218    12.347560975609632
    assert_close(4.6510228700066421, min(vol_atlas))
    assert_close(68.891611253882218, max(vol_atlas))

    vol = np.zeros((vol_atlas.size))
    for i in range(0, vol_atlas.size):
        vol[i] = vol_atlas[i] * pow(deg2rad, 2) * pow(radius, 2)
    # VOL(min/max):  57510668192.214096    851856184496.32886
    assert_close(57510668192.214096, min(vol))
    assert_close(851856184496.32886, max(vol))
    return as_field(vol, LocationType.Vertex)


def make_input_field(mesh, fs_nodes, edges_per_node):
    klevel = 0
    MXX = 0
    MYY = 1
    rpi = 2.0 * math.asin(1.0)
    radius = 6371.22e03
    deg2rad = 2.0 * rpi / 360.0

    zh0 = 2000.0
    zrad = 3.0 * rpi / 4.0 * radius
    zeta = rpi / 16.0 * radius
    zlatc = 0.0
    zlonc = 3.0 * rpi / 2.0

    m_rlonlatcr = fs_nodes.create_field(
        name="m_rlonlatcr", levels=1, dtype=np.float64, variables=edges_per_node
    )
    rlonlatcr = np.array(m_rlonlatcr, copy=False)

    m_rcoords = fs_nodes.create_field(
        name="m_rcoords", levels=1, dtype=np.float64, variables=edges_per_node
    )
    rcoords = np.array(m_rcoords, copy=False)

    m_rcosa = fs_nodes.create_field(name="m_rcosa", levels=1, dtype=np.float64)
    rcosa = np.array(m_rcosa, copy=False)

    m_rsina = fs_nodes.create_field(name="m_rsina", levels=1, dtype=np.float64)
    rsina = np.array(m_rsina, copy=False)

    m_pp = fs_nodes.create_field(name="m_pp", levels=1, dtype=np.float64)
    rzs = np.array(m_pp, copy=False)

    rcoords_deg = np.array(mesh.nodes.field("lonlat"))

    for jnode in range(0, fs_nodes.size):
        for i in range(0, 2):
            rcoords[jnode, klevel, i] = rcoords_deg[jnode, i] * deg2rad
            rlonlatcr[jnode, klevel, i] = rcoords[
                jnode, klevel, i
            ]  # This is not my pattern!
        rcosa[jnode, klevel] = math.cos(rlonlatcr[jnode, klevel, MYY])
        rsina[jnode, klevel] = math.sin(rlonlatcr[jnode, klevel, MYY])
    for jnode in range(0, fs_nodes.size):
        zlon = rlonlatcr[jnode, klevel, MXX]
        zdist = math.sin(zlatc) * rsina[jnode, klevel] + math.cos(zlatc) * rcosa[
            jnode, klevel
        ] * math.cos(zlon - zlonc)
        zdist = radius * math.acos(zdist)
        rzs[jnode, klevel] = 0.0
        if zdist < zrad:
            rzs[jnode, klevel] = rzs[jnode, klevel] + 0.5 * zh0 * (
                1.0 + math.cos(rpi * zdist / zrad)
            ) * math.pow(math.cos(rpi * zdist / zeta), 2)

    assert_close(0.0000000000000000, min(rzs))
    assert_close(1965.4980340735883, max(rzs))
    return as_field(rzs[:, klevel], LocationType.Vertex)


@stencil
def compute_zavgS(pp: Edge2Vertex, S_M):
    zavg = 0.5 * (pp[0] + pp[1])
    return S_M * zavg


@stencil
def compute_pnabla(
    pp: Tuple[Vertex2Edge, Edge2Vertex], S_M: Vertex2Edge, sign: Vertex2Edge, vol
):
    pnabla_M = 0
    zavgS = lift(compute_zavgS)(pp, S_M)
    for i in range(7):
        if zavgS[i] and sign[i]:
            pnabla_M += zavgS[i] * sign[i]
    return pnabla_M / vol


@stencil
def nabla(
    pp: Tuple[Vertex2Edge, Edge2Vertex],
    S_MXX: Vertex2Edge,
    S_MYY: Vertex2Edge,
    sign: Vertex2Edge,
    vol,
):
    return compute_pnabla(pp, S_MXX, sign, vol), compute_pnabla(pp, S_MYY, sign, vol)


def test_compute_zavgS():
    mesh, fs_edges, fs_nodes, edges_per_node = make_mesh()

    pp = make_input_field(mesh, fs_nodes, edges_per_node)
    S_MXX, S_MYY = make_S(mesh, fs_edges)

    edge_domain = list(range(fs_edges.size))

    zavgS = np.zeros((fs_edges.size))

    apply_stencil(
        compute_zavgS,
        [edge_domain],
        [
            make_connectivity_from_atlas(mesh.edges.node_connectivity, e2v),
        ],
        [zavgS],
        [pp, S_MXX],
    )
    assert_close(-199755464.25741270, min(zavgS))
    assert_close(388241977.58389181, max(zavgS))

    apply_stencil(
        compute_zavgS,
        [edge_domain],
        [
            make_connectivity_from_atlas(mesh.edges.node_connectivity, e2v),
        ],
        [zavgS],
        [pp, S_MYY],
    )
    assert_close(-1000788897.3202186, min(zavgS))
    assert_close(1000788897.3202186, max(zavgS))


def test_nabla():
    mesh, fs_edges, fs_nodes, edges_per_node = make_mesh()

    sign_acc = make_sign_field(mesh, fs_nodes.size, edges_per_node)  # acc of rank 1
    pp = make_input_field(mesh, fs_nodes, edges_per_node)
    S_MXX, S_MYY = make_S(mesh, fs_edges)
    vol = make_vol(mesh)

    nodes_domain = list(range(fs_nodes.size))

    pnabla_MXX = np.zeros((fs_nodes.size))
    pnabla_MYY = np.zeros((fs_nodes.size))
    apply_stencil(
        nabla,
        [nodes_domain],
        [
            make_connectivity_from_atlas(mesh.edges.node_connectivity, e2v),
            make_connectivity_from_atlas(mesh.nodes.edge_connectivity, v2e),
        ],
        [pnabla_MXX, pnabla_MYY],
        [pp, S_MXX, S_MYY, sign_acc, vol],
    )

    assert_close(-3.5455427772566003e-003, min(pnabla_MXX))
    assert_close(3.5455427772565435e-003, max(pnabla_MXX))
    assert_close(-3.3540113705465301e-003, min(pnabla_MYY))
    assert_close(3.3540113705465301e-003, max(pnabla_MYY))


@stencil
def nabla_from_sign_stencil(
    pp: Tuple[Vertex2Edge, Edge2Vertex],
    S_MXX: Vertex2Edge,
    S_MYY: Vertex2Edge,
    vol,
    node_index,
    pole_edges: Vertex2Edge,
    nodes_indices: Tuple[Vertex2Edge, Edge2Vertex],
):
    sign_acc = lift(sign_stencil)(node_index, pole_edges, nodes_indices)
    return compute_pnabla(pp, S_MXX, sign_acc, vol), compute_pnabla(
        pp, S_MYY, sign_acc, vol
    )


def test_nabla_from_sign_stencil():
    mesh, fs_edges, fs_nodes, edges_per_node = make_mesh()

    pp = make_input_field(mesh, fs_nodes, edges_per_node)
    S_MXX, S_MYY = make_S(mesh, fs_edges)
    vol = make_vol(mesh)

    edge_flags = np.array(mesh.edges.flags())
    pole_edges = as_field(
        np.array([Topology.check(flag, Topology.POLE) for flag in edge_flags]),
        LocationType.Edge,
    )
    index_field = as_field(np.array(range(fs_nodes.size)), LocationType.Vertex)

    nodes_domain = list(range(fs_nodes.size))

    pnabla_MXX = np.zeros((fs_nodes.size))
    pnabla_MYY = np.zeros((fs_nodes.size))
    apply_stencil(
        nabla_from_sign_stencil,
        [nodes_domain],
        [
            make_connectivity_from_atlas(mesh.edges.node_connectivity, e2v),
            make_connectivity_from_atlas(mesh.nodes.edge_connectivity, v2e),
        ],
        [pnabla_MXX, pnabla_MYY],
        [pp, S_MXX, S_MYY, vol, index_field, pole_edges, index_field],
    )

    assert_close(-3.5455427772566003e-003, min(pnabla_MXX))
    assert_close(3.5455427772565435e-003, max(pnabla_MXX))
    assert_close(-3.3540113705465301e-003, min(pnabla_MYY))
    assert_close(3.3540113705465301e-003, max(pnabla_MYY))


if __name__ == "__main__":
    test_nabla_from_sign_stencil()
