import numpy as np

from gt4py_new_model.storage import storage, index


def test_storage():
    rng = np.random.default_rng()
    array = rng.normal(size=(2, 3))
    s = storage(array)
    assert np.all(np.asarray(s) == array)


def test_index():
    i, j = np.indices((2, 3))
    i_s = index((2, 3), "i")
    j_s = index((2, 3), "j")
    assert np.all(np.asarray(i_s) == i)
    assert np.all(np.asarray(j_s) == j)
