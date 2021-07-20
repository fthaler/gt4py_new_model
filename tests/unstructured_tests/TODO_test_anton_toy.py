from unstructured.builtins import *
from unstructured.embedded import np_as_located_field
from unstructured.runtime import *
import numpy as np


@fundef
def ldif(d):
    return lambda inp: deref(shift(d, -1)(inp)) - deref(inp)


@fundef
def rdif(d):
    return compose(ldif(d), shift(d, 1))


@fundef
def dif2(d):
    return compose(ldif(d), lift(rdif(d)))


i = offset("i")
j = offset("j")


@fundef
def lap(inp):
    return dif2(i)(inp) + dif2(j)(inp)


IDim = CartesianAxis("IDim")
JDim = CartesianAxis("JDim")
KDim = CartesianAxis("KDim")


@fendef
def fencil(x, y, z, output, input):
    closure(
        domain(
            named_range(IDim, 0, x), named_range(JDim, 0, y), named_range(KDim, 0, z)
        ),
        lap,
        [output],
        [input],
    )


fencil(*([None] * 5), backend="lisp")
fencil(*([None] * 5), backend="cpptoy")


def test_anton_toy():
    shape = [5, 7, 9]
    rng = np.random.default_rng()
    inp = np_as_located_field(IDim, JDim, KDim, origin={IDim: 1, JDim: 1, KDim: 0})(
        rng.normal(size=(shape[0] + 2, shape[1] + 2, shape[2])),
    )
    out = np_as_located_field(IDim, JDim, KDim)(np.zeros(shape))

    # ref = TODO

    fencil(
        shape[0],
        shape[1],
        shape[2],
        out,
        inp,
        backend="double_roundtrip",
        offset_provider={"i": IDim, "j": JDim},
    )

    # assert np.allclose(out, ref)
