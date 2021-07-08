from inspect import Traceback
from unstructured.runtime import offset
from unstructured.tracing import (
    OffsetLiteral,
    Tracer,
    apply_stencil,
    cartesian,
    compose,
    # deref,
    # fendef,
    # fundef,
    # lift,
    # shift,
)

from unstructured.builtins import deref, shift, lift
from unstructured.runtime import fundef, fendef, closures
import unstructured
from unstructured import tracing
from devtools import debug


@fundef
def ldif(d):
    return lambda inp: deref(shift(d, -1)(inp)) - deref(inp)


@fundef
def rdif(d):
    return compose(ldif(d), shift(d, 1))


@fundef
def dif2(d):
    return compose(ldif(d), lift(rdif(d)))


# i = offset("i")
# j = offset("j")
i = OffsetLiteral(value="i")
j = OffsetLiteral(value="j")


@fundef
def lap(inp):
    return dif2(i)(inp) + dif2(j)(inp)


@fendef
def testee(xs, xe, ys, ye, z, output, input):
    return closures(
        apply_stencil(cartesian(xs, xe, ys, ye, 0, z), lap, [output], [input])
    )


testee(*([None] * 7), backend="lisp")
testee(*([None] * 7), backend="cpptoy")
