from unstructured.builtins import deref, shift, lift, compose, cartesian
from unstructured.runtime import fundef, fendef, closure, offset
from devtools import debug


@fundef
def lap(inp):
    return deref(inp) + 3


@fendef
def testee(xs, xe, ys, ye, z, output, input):
    closure(cartesian(xs, xe, ys, ye, 0, z), lap, [output], [input])


testee(*([None] * 7), backend="lisp")
testee(*([None] * 7), backend="cpptoy")
