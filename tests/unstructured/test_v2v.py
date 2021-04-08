from dataclasses import dataclass
import numpy as np
import math
import enum


def as_1d(arr):
    return arr.flatten()


def as_2d(arr, shape):
    return arr.reshape(*shape)


def connectivity(*args):
    def _impl(fun):
        class conn:
            # TODO NeighborHood not only in and out location
            in_location, out_location = args[0], args[1]

            def __call__(self, field):
                assert hasattr(field, "location")
                assert field.location == self.in_location

                return fun(field)

        return conn()

    return _impl


def make_v2v_conn(shape_2d):
    strides = [shape_2d[1], 1]

    @connectivity(LocationType.Vertex, LocationType.Vertex)
    def v2v_conn(field):
        class new_field:
            location = LocationType.Vertex  # TODO information is duplicated

            def __call__(self, index):
                return [
                    field(index + strides[0]),
                    field(index - strides[0]),
                    field(index + strides[1]),
                    field(index - strides[1]),
                ]

        return new_field()

    return v2v_conn


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


class V2VNeighborHood:
    left = 0
    right = 1
    top = 2
    bottom = 3


vv = V2VNeighborHood()


def stencil(*args):
    def decorator_stencil(fun):
        return fun

    return decorator_stencil


@stencil([vv])
def v2v(acc_in):
    return acc_in[vv.left] + acc_in[vv.right] + acc_in[vv.top] + acc_in[vv.bottom]


# @stencil([neighborhood, neighborhood])
def v2v2v(acc_in):
    x = lift(v2v)(acc_in)
    return v2v(x)


def test_v2v():
    # TODO define fencil
    def apply(stencil, domain, v2v_conn, out, inp):
        for i in domain:
            out[i] = stencil(v2v_conn(as_field(inp, LocationType.Vertex))(i))

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
    apply(v2v, inner_domain, v2v_conn, out1d, inp1d)
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
    apply(v2v2v, inner_domain, v2v_conn, out1d, inp1d)
    out2d = as_2d(out1d, shape)
    assert np.allclose(out2d, ref)


test_v2v()
test_v2v2v()
