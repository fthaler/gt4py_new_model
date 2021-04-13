from unstructured.concepts import (
    ufield,
    LocationType,
    conn_mult,
    connectivity,
    neighborhood,
)


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
    assert dummy_e2v_conn.in_location == LocationType.Edge
    assert dummy_e2v_conn.out_location == LocationType.Vertex

    e2v2e = conn_mult(dummy_e2v_conn, dummy_v2e_conn)
    assert e2v2e.in_location == LocationType.Edge
    assert e2v2e.out_location == LocationType.Edge

    e2v2e2v = conn_mult(e2v2e, dummy_e2v_conn)
    assert e2v2e2v.in_location == LocationType.Edge
    assert e2v2e2v.out_location == LocationType.Vertex
