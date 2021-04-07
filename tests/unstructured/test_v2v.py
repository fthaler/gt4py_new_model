import numpy as np
import math


def as_1d(arr):
    return arr.flatten()


def as_2d(arr, shape):
    return arr.reshape(*shape)


def v2v(acc_in):
    print(acc_in[0])
    return acc_in[0] + acc_in[1] + acc_in[2] + acc_in[3]


def make_v2v_conn(shape_2d):
    strides = [shape_2d[1], 1]

    def v2v_conn(field):
        def new_field(index):
            return [
                field(index + strides[0]),
                field(index - strides[0]),
                field(index + strides[1]),
                field(index - strides[1]),
            ]

        return new_field

    return v2v_conn


def lift(stencil):
    def lifted(acc):
        class wrap:
            def __getitem__(self, i):
                return stencil(acc[i])

        return wrap()

    return lifted


def v2v2v(acc_in):
    x = lift(v2v)(acc_in)
    return v2v(x)


# a field is a function from index to element `()` not `[]`
# (or change the conn)
def as_field(arr):
    def tmp(i):
        return arr[i]

    return tmp


# TODO define fencil
def apply(stencil, domain, v2v_conn, out, inp):
    for i in domain:
        out[i] = stencil(v2v_conn(v2v_conn(as_field(inp)))(i))


def test_copy():
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


test_copy()
