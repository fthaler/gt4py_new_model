import numpy as np
import math

from unstructured.concepts import (
    LocationType,
    apply_stencil,
    neighborhood,
    connectivity,
    stencil,
    lift,
    ufield,
)
from unstructured.helpers import as_1d, as_2d, as_field


@neighborhood(LocationType.Vertex, LocationType.Vertex)
class V2VNeighborHood:
    left = 0
    right = 1
    top = 2
    bottom = 3


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


def test_v2v():
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

    apply_stencil(
        v2v, inner_domain, [v2v_conn], out1d, as_field(inp1d, LocationType.Vertex)
    )
    out2d = as_2d(out1d, shape)
    assert np.allclose(out2d, ref)


def test_v2v2v():
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
