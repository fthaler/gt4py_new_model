from unstructured.builtins import *
from unstructured.runtime import *


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


@fendef
def testee(xs, xe, ys, ye, z, output, input):
    closure(cartesian(xs, xe, ys, ye, 0, z), lap, [output], [input])


testee(*([None] * 7), backend="lisp")
testee(*([None] * 7), backend="cpptoy")
testee(*([None] * 7), backend="embedded")
