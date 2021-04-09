from dataclasses import dataclass
import numpy as np
import math
import devtools
import enum


def as_1d(arr):
    return arr.flatten()


def as_2d(arr, shape):
    return arr.reshape(*shape)


def connectivity(*_neighborhoods):
    def _impl(fun):
        class conn:
            # TODO NeighborHood not only in and out location
            neighborhoods = _neighborhoods
            in_location, out_location = (
                neighborhoods[0].in_location,
                neighborhoods[-1].out_location,
            )

            def __call__(self, field):
                assert hasattr(field, "location")
                assert field.location == self.in_location

                return fun(field)

        return conn()

    return _impl


def conn_mult(conn_a, conn_b):
    assert conn_a.neighborhoods[-1].out_location == conn_b.neighborhoods[0].in_location

    class conn:
        neighborhoods = [*conn_a.neighborhoods, *conn_b.neighborhoods]
        in_location, out_location = (
            neighborhoods[0].in_location,
            neighborhoods[-1].out_location,
        )

        def __call__(self, field):
            return conn_a(conn_b(field))

    return conn()


def stencil(*args):
    def _impl(fun):
        class decorated_stencil:
            acc_neighborhood_chains = args

            def __call__(self, *args):
                return fun(*args)

        return decorated_stencil()

    return _impl


def lift(stencil):
    def lifted(acc):
        class wrap:
            def __getitem__(self, i):
                return stencil(acc[i])

        return wrap()

    return lifted


@enum.unique
class LocationType(enum.IntEnum):
    Vertex = 0
    Edge = 1
    Cell = 2


# a field is a function from index to element `()` not `[]`
# (or change the conn)
def as_field(arr, loc: LocationType):
    class _field:
        location = loc

        def __call__(self, i: int):
            return arr[i]

    return _field()


def neighborhood(in_loc, out_loc):
    """syntactic sugar to create a neighborhood"""

    def impl(cls):
        cls.in_location = in_loc
        cls.out_location = out_loc
        return cls

    return impl


@neighborhood(LocationType.Vertex, LocationType.Vertex)
class V2VNeighborHood:
    left = 0
    right = 1
    top = 2
    bottom = 3


def ufield(loc):
    def _impl(fun):
        class _field:
            location = loc

            def __call__(self, index):
                return fun(index)

        return _field()

    return _impl


def make_v2v_conn(shape_2d):
    strides = [shape_2d[1], 1]

    @connectivity(V2VNeighborHood())
    def v2v_conn(field):
        @ufield(LocationType.Vertex)
        def _field(index):
            return [
                field(index + strides[0]),
                field(index - strides[0]),
                field(index + strides[1]),
                field(index - strides[1]),
            ]

        return _field

    return v2v_conn


vv = V2VNeighborHood()


@stencil((vv,))
def v2v(acc_in):
    return acc_in[vv.left] + acc_in[vv.right] + acc_in[vv.top] + acc_in[vv.bottom]


@stencil((vv, vv))
def v2v2v(acc_in):
    x = lift(v2v)(acc_in)
    return v2v(x)


def neighborhood_chain_to_key(neighborhood_chain):
    return tuple(type(nh).__name__ for nh in neighborhood_chain)


def neighborhood_to_key(neighborhood):
    return type(neighborhood).__name__


def apply_stencil(
    stencil, domain, connectivities, out, inp
):  # TODO support multiple input/output
    conn_map = {}
    for c in connectivities:
        assert (
            len(c.neighborhoods) == 1
        )  # we assume the user passes only non-multiplied connectivities (just for simplification)
        conn_map[neighborhood_to_key(c.neighborhoods[0])] = c

    cur_conn = None
    assert len(stencil.acc_neighborhood_chains) == 1  # TODO only one argument supported
    for nh in stencil.acc_neighborhood_chains[0]:
        print(nh)
        if not cur_conn:
            cur_conn = conn_map[neighborhood_to_key(nh)]
        else:
            cur_conn = conn_mult(cur_conn, conn_map[neighborhood_to_key(nh)])
    stencil_args = cur_conn(inp)

    for i in domain:
        out[i] = stencil(stencil_args(i))


@neighborhood(LocationType.Edge, LocationType.Vertex)
class E2VNeighborHood:
    pass


@neighborhood(LocationType.Vertex, LocationType.Edge)
class V2ENeighborHood:
    pass


@connectivity(E2VNeighborHood())
def dummy_e2v_conn(field):
    class new_field:
        location = LocationType.Edge

        def __call__(self, index):
            return []

    return new_field()


@connectivity(V2ENeighborHood())
def dummy_v2e_conn(field):
    class new_field:
        location = LocationType.Vertex

        def __call__(self, index):
            return []

    return new_field()


def test_conn_multiply():
    v2vconn = make_v2v_conn((5, 7))
    v2v2vconn = conn_mult(v2vconn, v2vconn)
    print(v2v2vconn.neighborhoods)

    print(conn_mult(dummy_e2v_conn, dummy_v2e_conn).neighborhoods)


test_conn_multiply()


def test_v2v():
    # TODO define fencil
    # def apply(stencil, domain, v2v_conn, out, inp):
    #     for i in domain:
    #         out[i] = stencil(v2v_conn(as_field(inp, LocationType.Vertex))(i))

    shape = (5, 7)
    # inp = np.random.rand(*shape)
    inp = np.ones(shape)
    out1d = np.zeros(math.prod(shape))
    ref = np.zeros(shape)
    ref[1:-1, 1:-1] = np.ones((3, 5)) * 4

    inp1d = as_1d(inp)

    domain = np.arange(math.prod(shape))
    domain_2d = as_2d(domain, shape)
    inner_domain = as_1d(domain_2d[1:-1, 1:-1])

    v2v_conn = make_v2v_conn(shape)

    print(v2v_conn.in_location)
    # apply(v2v, inner_domain, v2v_conn, out1d, inp1d)
    apply_stencil(
        v2v, inner_domain, [v2v_conn], out1d, as_field(inp1d, LocationType.Vertex)
    )
    out2d = as_2d(out1d, shape)
    assert np.allclose(out2d, ref)


def test_v2v2v():
    # TODO define fencil
    def apply(stencil, domain, v2v_conn, out, inp):
        for i in domain:
            out[i] = stencil(v2v_conn(v2v_conn(as_field(inp, LocationType.Vertex)))(i))

    shape = (5, 7)
    # inp = np.random.rand(*shape)
    inp = np.ones(shape)
    out1d = np.zeros(math.prod(shape))
    ref = np.zeros(shape)
    ref[2:-2, 2:-2] = np.ones((1, 3)) * 16

    inp1d = as_1d(inp)

    domain = np.arange(math.prod(shape))
    domain_2d = as_2d(domain, shape)
    inner_domain = as_1d(domain_2d[2:-2, 2:-2])

    v2v_conn = make_v2v_conn(shape)
    apply_stencil(
        v2v2v, inner_domain, [v2v_conn], out1d, as_field(inp1d, LocationType.Vertex)
    )
    out2d = as_2d(out1d, shape)
    assert np.allclose(out2d, ref)


test_v2v()
test_v2v2v()
