from unstructured.builtins import *
from unstructured.runtime import *


@fundef
def lap(inp):
    return deref(inp) + 3


I = CartesianAxis("I")
J = CartesianAxis("J")
K = CartesianAxis("K")


@fendef
def testee(xs, xe, ys, ye, z, output, input):
    closure(
        domain(
            named_range(I, xs, xe),
            named_range(J, ys, ye),
            named_range(K, 0, z),
        ),
        lap,
        [output],
        [input],
    )


testee(*([None] * 7), backend="lisp")
testee(*([None] * 7), backend="cpptoy")
testee(*([None] * 7), backend="embedded")
