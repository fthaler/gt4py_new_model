import numpy as np
import pytest
from unstructured.helpers import (
    array_as_field,
    index_field,
    print_dims,
)
from unstructured.utils import (
    Dimension,
    dimensions_compatible,
    remove_axis,
    remove_indices_of_axises,
    axis,
)


@axis()
class Edge:
    pass


@axis()
class Vertex:
    pass


@axis(length=2)
class E2V:
    pass


@axis(length=4)
class V2E:
    pass


@pytest.fixture
def vertex_field():
    return array_as_field(Vertex)(np.ones([9]))


@pytest.fixture
def edge_field():
    return array_as_field(Edge)(np.ones([18]))


@pytest.fixture
def edge_index_field():
    return index_field(Edge)


@pytest.fixture
def vertex_index_field():
    return index_field(Vertex)


def test_index_field(edge_index_field):
    assert all([i == edge_index_field[Edge(i)] for i in range(18)])


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

e2v_neightbl = array_as_field(Edge, E2V, element_type=Vertex)(e2v_arr)

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

v2e_neightbl = array_as_field(Vertex, V2E, element_type=Edge)(v2e_arr)


def test_neightbl():
    vertex_index_field = index_field(Vertex)
    e2v_field = vertex_index_field[e2v_neightbl]
    assert e2v_field[Edge(4), E2V(1)] == 5


# test_neightbl()


def test_element_type():
    assert v2e_neightbl.element_type == Edge
    assert e2v_neightbl.element_type == Vertex
    assert isinstance(e2v_neightbl[Edge(1), E2V(0)], Vertex)
    assert isinstance(v2e_neightbl[Vertex(1), V2E(0)], Edge)


def test_remove_axis():
    axises = (Edge, Vertex, E2V)

    assert remove_axis(Edge, axises) == (Vertex, E2V)
    assert remove_axis(Vertex, axises) == (Edge, E2V)
    assert remove_axis(E2V, axises) == (Edge, Vertex)


def test_remove_indices():
    removed = remove_indices_of_axises(
        (Edge, Vertex), (Edge(0), V2E(1), Vertex(2), E2V(3))
    )

    assert removed[0] == V2E(1)
    print(len(removed))
    print(removed[0])
    print(removed[1])


# Connectivity LocA2LocB means give me neighbors of type LocB from LocA,
# i.e. it takes Field[LocB] and returns Field[LocA, LocA2LocB]
# neigh_tbl is Field[LocA, LocA2LocB] where elements are the indices in LocB
# def make_connectivity(neigh_loc, neigh_tbl):
#     def _conn(field):
#         if neigh_loc not in field.axises:
#             raise TypeError("Incompatible field passed to connectivity.")

#         @element_access_to_field(
#             axises=remove_axis(neigh_loc, field.axises) + neigh_tbl.axises,
#             element_type=field.element_type,
#         )
#         def element_access(indices):
#             field_index = neigh_tbl[
#                 get_index_of_type(neigh_tbl.axises[0])(indices),
#                 get_index_of_type(neigh_tbl.axises[1])(indices),
#             ]

#             new_indices = (field_index,) + remove_indices_of_axises(
#                 (neigh_tbl.axises[0], neigh_tbl.axises[1]), indices
#             )

#             return field[new_indices]

#         return element_access

#     return _conn


def make_connectivity(_, neigh_tbl):
    def _conn(field):
        return field[neigh_tbl]

    return _conn


@pytest.fixture
def e2v():
    return make_connectivity(Vertex, e2v_neightbl)


@pytest.fixture
def v2e():
    return make_connectivity(Edge, v2e_neightbl)


def test_field_and_connectivity_compatible(e2v, vertex_field, edge_field):
    e2v_field = e2v(vertex_field)
    assert dimensions_compatible(
        e2v_field.dimensions, (Dimension(Edge, None), Dimension(E2V, None))
    )

    with pytest.raises(TypeError):
        e2v(edge_field)


def test_slicing():
    vertex_edge_e2v_field = array_as_field(Vertex, Edge, E2V)(np.ones((9, 18, 2)))

    print_dims(vertex_edge_e2v_field)
    assert dimensions_compatible(
        vertex_edge_e2v_field.dimensions,
        (
            Dimension(Vertex, range(9)),
            Dimension(Edge, range(18)),
            Dimension(E2V, range(2)),
        ),
    )
    assert dimensions_compatible(
        vertex_edge_e2v_field[Vertex(0)].dimensions,
        (Dimension(Edge, None), Dimension(E2V, None)),
    )

    assert dimensions_compatible(
        vertex_edge_e2v_field[Vertex(0)][E2V(0)].dimensions, (Dimension(Edge, None),)
    )
    assert dimensions_compatible(
        vertex_edge_e2v_field[Vertex(0), Edge(0)].dimensions, (Dimension(E2V, None),)
    )

    assert vertex_edge_e2v_field[Vertex(0)][Edge(0), E2V(0)] == 1

    field_sum = vertex_edge_e2v_field + vertex_edge_e2v_field
    assert field_sum[Vertex(0), Edge(0), E2V(0)] == 2


def test_connectivity(e2v, v2e, vertex_index_field):
    v_of_e = e2v(vertex_index_field)
    assert v_of_e[Edge(6), E2V(0)] == 6
    assert v_of_e[Edge(6), E2V(1)] == 7
    assert v_of_e[Edge(16), E2V(0)] == 7
    assert v_of_e[Edge(16), E2V(1)] == 1

    v_of_e_of_v = v2e(v_of_e)
    assert v_of_e_of_v[Vertex(6), V2E(0), E2V(0)] == 6
    assert v_of_e_of_v[Vertex(6), V2E(1), E2V(0)] == 3
    assert v_of_e_of_v[Vertex(4), V2E(3), E2V(1)] == 7

    v_of_e_of_v_of_e = e2v(v_of_e_of_v)
    assert v_of_e_of_v_of_e[Edge(4), E2V(1), V2E(0), E2V(0)] == 5  # order matters!
