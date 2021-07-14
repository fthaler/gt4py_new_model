from unstructured.builtins import *
from unstructured.runtime import *

from devtools import debug


@fundef
def lap(inp):
    return deref(inp) + 3


@fundef
def indirect(inp):
    return deref(lift(lap)(inp))


@fendef
def testee(xs, xe, ys, ye, z, output, input):
    closure(cartesian(xs, xe, ys, ye, 0, z), indirect, [output], [input])


testee(*([None] * 7), backend="lisp")
testee(*([None] * 7), backend="cpptoy")
