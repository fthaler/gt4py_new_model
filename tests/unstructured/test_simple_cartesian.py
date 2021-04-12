import math
import numpy as np
import pytest

from unstructured.concepts import (
    LocationType,
    apply_stencil,
    lift,
    neighborhood,
    stencil,
)
from unstructured.helpers import as_1d, as_2d, as_field, simple_connectivity


@neighborhood(LocationType.Vertex, LocationType.Vertex)
class CartesianNeighborHood:
    pass


cart = CartesianNeighborHood()


@simple_connectivity(cart)
def cartesian_connectivity_1d(index):
    class neighs:
        def __getitem__(self, offset):
            return index + offset

    return neighs()


@stencil((cart,))
def laplacian1d(inp):
    return -2 * inp[0] + (inp[-1] + inp[1])


def test_lap1d():
    shape = 10
    inp = np.arange(shape) * np.arange(shape)

    out = np.zeros(shape)
    domain = np.arange(1, shape - 1)

    apply_stencil(
        laplacian1d,
        domain,
        [cartesian_connectivity_1d],
        out,
        [as_field(inp, LocationType.Vertex)],
    )

    ref = np.zeros(shape)
    ref[1:-1] = 2

    assert np.allclose(out, ref)


test_lap1d()
