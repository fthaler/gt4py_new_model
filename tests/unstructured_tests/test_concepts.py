import numpy as np
from unstructured.concepts import apply_stencil
from unstructured.helpers import (
    array_as_field,
)
from unstructured.utils import (
    axis,
)


@axis()
class Vertex:
    pass


def test_or():
    mask1 = array_as_field(Vertex)(np.array([True, False, False, True]))
    mask2 = array_as_field(Vertex)(np.array([True, True, False, False]))

    ref = np.array([True, True, False, True])

    or_ = mask1 | mask2

    out = np.ndarray((4,), bool)
    apply_stencil(lambda: or_, [(range(4), Vertex)], [], [out])

    print(out[0])
    print(out[1])
    print(out[2])
    print(out[3])

    assert (out == ref).all()
