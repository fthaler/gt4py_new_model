import numpy as np

from gt4py_new_model.accessors import ArrayColumnAccessor
from gt4py_new_model.dimensions import K
from gt4py_new_model.builtins import if_then_else, scan


def test_if_then_else():
    assert if_then_else(True, 1, 2) == 1
    assert if_then_else(False, 1, 2) == 2

    assert np.all(
        if_then_else(np.array([True, False]), np.array([1, 2]), np.array([3, 4]))
        == np.array([1, 4])
    )

    assert np.all(
        if_then_else(True, np.array([1, 2]), np.array([3, 4])) == np.array([1, 2])
    )


def test_scan():
    def acc(column):
        return ArrayColumnAccessor(
            array=np.array(column),
            dimensions=("k",),
            focus=(0,),
            column="k",
            size=len(column),
        )

    def identity(state, x):
        return x[K]

    assert np.all(scan(identity, True, 1, acc([1, 2])) == np.array([1, 2]))
    assert np.all(scan(identity, False, 1, acc([1, 2])) == np.array([1, 2]))

    def colsum(state, x):
        return state + x[K]

    assert np.all(scan(colsum, True, 1, acc([1, 2])) == np.array([2, 4]))
    assert np.all(scan(colsum, False, 1, acc([1, 2])) == np.array([4, 3]))

    def tup(state, x):
        return state[0] + x[K], state[1] - x[K]

    a, b = scan(tup, True, (1, 1), acc([1, 2]))
    assert np.all(a == np.array([2, 4]))
    assert np.all(b == np.array([0, -2]))
