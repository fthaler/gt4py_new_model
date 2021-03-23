import numpy as np

from gt4py_new_model.accessors import (
    array_column_accessor,
    index_column_accessor,
)
from gt4py_new_model.dimensions import Dimension


def test_column_accessor():
    rng = np.random.default_rng(42)
    array = rng.normal(size=(11, 9, 3))
    accessor = array_column_accessor(
        array=array, dimensions=("a", "b", "c"), focus=(1, 3, 2), column="b", size=4
    )

    a = Dimension("a")
    b = Dimension("b")
    c = Dimension("c")
    d = Dimension("d")
    assert np.all(accessor[d] == array[1, 3:7, 2])
    assert np.all(accessor[a, b, c] == array[1, 3:7, 2])
    assert np.all(accessor[a + 1] == array[2, 3:7, 2])
    assert np.all(accessor[b - 1, c - 2] == array[1, 2:6, 0])
    assert np.all(accessor[b - 5, c - 2][2:] == array[1, 0:2, 0])


def test_index_column_accessor():
    accessor = index_column_accessor(dimension="a", focus=3, column="a", size=2)
    a = Dimension("a")
    b = Dimension("b")
    assert np.all(accessor[a] == np.arange(3, 5))
    assert np.all(accessor[a + 1] == np.arange(4, 6))
    assert np.all(accessor[b + 1] == np.arange(3, 5))
    accessor = index_column_accessor(dimension="a", focus=3, column="b", size=2)
    a = Dimension("a")
    b = Dimension("b")
    assert np.all(accessor[a] == np.full(2, 3))
    assert np.all(accessor[a + 1] == np.full(2, 4))
    assert np.all(accessor[b + 1] == np.full(2, 3))
