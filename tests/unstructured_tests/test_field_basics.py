import numpy as np
from unstructured.utils import axis
from unstructured.helpers import array_as_field


@axis()
class Dim:
    pass


def test_addition():
    field1 = array_as_field(Dim)(np.array([1, 2, 3, 4]))
    field2 = array_as_field(Dim)(np.array([1, 2, 3, 4]))

    res = field1 + field2

    assert res.dimensions[0].range == range(0, 4)
    assert res[Dim(2)] == 6

    ref = np.array([2, 4, 6, 8], float)

    assert np.allclose(ref, res)


test_addition()


@axis()
class Dim2:
    pass


def test_dimension_swap():
    field = array_as_field(Dim, Dim2)(np.zeros((4, 5)))

    out_field = np.zeros((5, 4))

    out_field[:] = field.array_of(Dim2, Dim)