import numpy as np
import math

from unstructured.concepts import LocationType, apply_stencil, stencil
from unstructured.helpers import as_field, as_1d, as_2d


@stencil(())
def copy(acc_in):
    return acc_in


def test_copy():
    shape = (5, 7)
    inp = np.random.rand(*shape)
    out1d = np.zeros(math.prod(shape))

    inp1d = as_1d(inp)

    domain = list(range(math.prod(shape)))

    apply_stencil(copy, [domain], [], out1d, [as_field(inp1d, LocationType.Vertex)])
    out2d = as_2d(out1d, shape)
    assert np.allclose(out2d, inp)


test_copy()
