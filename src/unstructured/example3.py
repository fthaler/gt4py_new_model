from unstructured.builtins import *
from unstructured.runtime import *

from unstructured.embedded import np_as_located_field
import numpy as np

I = offset("I")
J = offset("J")
I_loc = CartesianAxis("I_loc")
J_loc = CartesianAxis("J_loc")


@fundef
def foo(inp):
    return deref(shift(J, 1)(inp))


@fendef(offset_provider={"I": I_loc, "J": J_loc})
def testee(output, input):
    closure(
        cartesian(cartesian_range(I_loc, 0, 1), cartesian_range(J_loc, 0, 1)),
        foo,
        [output],
        [input],
    )


testee(None, None, backend="cpptoy")


@fendef(offset_provider={"I": J_loc, "J": I_loc})
def testee_swapped(output, input):
    closure(
        cartesian(cartesian_range(I_loc, 0, 1), cartesian_range(J_loc, 0, 1)),
        foo,
        [output],
        [input],
    )


testee(*([None] * 2), backend="lisp")
testee(*([None] * 2), backend="cpptoy")


inp = np_as_located_field(I_loc, J_loc)(np.asarray([[0, 42], [1, 43]]))
out = np_as_located_field(I_loc, J_loc)(np.asarray([[-1]]))
testee(out, inp)
print(out[0][0])

testee_swapped(out, inp)
print(out[0][0])

testee(out, inp)
print(out[0][0])

testee(out, inp, backend="embedded")
print(out[0, 0])


@fundef
def foo2(inp):
    return deref(shift(I, J, 1, 1)(inp))


@fendef(offset_provider={"I": J_loc, "J": I_loc})
def testee2(output, input):
    closure(
        cartesian(cartesian_range(I_loc, 0, 1), cartesian_range(J_loc, 0, 1)),
        foo2,
        [output],
        [input],
    )


testee2(out, inp)
print(out[0, 0])

testee2(out, inp, backend="embedded", debug=True)
print(out[0, 0])
