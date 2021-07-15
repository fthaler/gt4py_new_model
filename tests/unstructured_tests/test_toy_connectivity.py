import numpy as np
from numpy.core.numeric import allclose
from unstructured.runtime import *
from unstructured.builtins import *
from unstructured.embedded import (
    NeighborTableOffsetProvider,
    np_as_located_field,
    index_field,
)


Vertex = CartesianAxis("Vertex")
Edge = CartesianAxis("Edge")


# 3x3 periodic   edges
# 0 - 1 - 2 -    0 1 2
# |   |   |      9 10 11
# 3 - 4 - 5 -    3 4 5
# |   |   |      12 13 14
# 6 - 7 - 8 -    6 7 8
# |   |   |      15 16 17

e2v_arr = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 0],
        [3, 4],
        [4, 5],
        [5, 3],
        [6, 7],
        [7, 8],
        [8, 6],
        [0, 3],
        [1, 4],
        [2, 5],
        [3, 6],
        [4, 7],
        [5, 8],
        [6, 0],
        [7, 1],
        [8, 2],
    ]
)


# order east, north, west, south (counter-clock wise)
v2e_arr = np.array(
    [
        [0, 15, 2, 9],  # 0
        [1, 16, 0, 10],
        [2, 17, 1, 11],
        [3, 9, 5, 12],  # 3
        [4, 10, 3, 13],
        [5, 11, 4, 14],
        [6, 12, 8, 15],  # 6
        [7, 13, 6, 16],
        [8, 14, 7, 17],
    ]
)

V2E = offset("V2E")


@fundef
def sum_edges_to_vertices(in_edges):
    return (
        deref(shift(V2E, 0)(in_edges))
        + deref(shift(V2E, 1)(in_edges))
        + deref(shift(V2E, 2)(in_edges))
        + deref(shift(V2E, 3)(in_edges))
    )


@fendef(offset_provider={"V2E": NeighborTableOffsetProvider(v2e_arr, Vertex, Edge, 4)})
def e2v_sum_fencil(in_edges, out_vertices):
    closure(
        domain(named_range(Vertex, 0, 9)),
        sum_edges_to_vertices,
        [out_vertices],
        [in_edges],
    )


def test_sum_edges_to_vertices():
    inp = index_field(Edge)
    out = np_as_located_field(Vertex)(np.zeros([9]))
    ref = np.asarray(list(sum(row) for row in v2e_arr))

    e2v_sum_fencil(inp, out, backend="double_roundtrip")
    assert allclose(out, ref)
    e2v_sum_fencil(None, None, backend="cpptoy")


@fundef
def sum_edges_to_vertices_reduce(in_edges):
    return reduce(lambda a, b: a + b, 0)(shift(V2E)(in_edges))


@fendef(offset_provider={"V2E": NeighborTableOffsetProvider(v2e_arr, Vertex, Edge, 4)})
def e2v_sum_fencil_reduce(in_edges, out_vertices):
    closure(
        domain(named_range(Vertex, 0, 9)),
        sum_edges_to_vertices_reduce,
        [out_vertices],
        [in_edges],
    )


def test_sum_edges_to_vertices_reduce():
    inp = index_field(Edge)
    out = np_as_located_field(Vertex)(np.zeros([9]))
    ref = np.asarray(list(sum(row) for row in v2e_arr))

    e2v_sum_fencil_reduce(None, None, backend="cpptoy")
    e2v_sum_fencil_reduce(inp, out)
    # e2v_sum_fencil_reduce(inp, out, backend="double_roundtrip")
    assert allclose(out, ref)


@fundef
def sparse_stencil(inp):
    return reduce(lambda a, b: a + b, 0)(inp)


@fendef(offset_provider={"V2E": NeighborTableOffsetProvider(v2e_arr, Vertex, Edge, 4)})
def sparse_fencil(inp, out):
    closure(
        domain(named_range(Vertex, 0, 9)),
        sparse_stencil,
        [out],
        [inp],
    )


def test_sparse_input_field():
    inp = np_as_located_field(Vertex, V2E)(np.asarray([[1, 2, 3, 4]] * 9))
    out = np_as_located_field(Vertex)(np.zeros([9]))

    ref = np.ones([9]) * 10

    sparse_fencil(inp, out, backend="double_roundtrip")

    assert allclose(out, ref)
