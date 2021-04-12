import numpy as np
from numpy.core.numeric import outer

from unstructured.concepts import (
    LocationType,
    apply_stencil,
    neighborhood,
    stencil,
)
from unstructured.helpers import as_field, simple_connectivity


@neighborhood(LocationType.Vertex, LocationType.Vertex)
class CartesianNeighborHood:
    pass


cart = CartesianNeighborHood()


@simple_connectivity(cart)
def cartesian_connectivity(*indices):
    class neighs:
        def __getitem__(self, neighindices):
            if not isinstance(neighindices, tuple):
                neighindices = (neighindices,)
            return tuple(
                map(lambda x: x[0] + x[1], zip(indices, neighindices)),
            )

    return neighs()


@stencil((cart,))
def laplacian1d(inp):
    return -2 * inp[0] + (inp[-1] + inp[1])


def test_lap1d():
    shape = 10
    inp = np.arange(shape) * np.arange(shape)

    out = np.zeros(shape)
    domain = list(range(1, shape - 1))

    apply_stencil(
        laplacian1d,
        [domain],
        [cartesian_connectivity],
        out,
        [as_field(inp, LocationType.Vertex)],
    )

    ref = np.zeros(shape)
    ref[1:-1] = 2

    assert np.allclose(out, ref)


test_lap1d()


@stencil((cart,))
def laplacian2d(inp):
    return -4 * inp[0, 0] + (inp[-1, 0] + inp[1, 0] + inp[0, -1] + inp[0, 1])


def test_lap():
    shape = (5, 7)
    inp = np.zeros(shape)
    inp[:, :] = np.arange(shape[1]) * np.arange(shape[1])

    out = np.zeros(shape)
    domain = [list(range(1, shape[0] - 1)), list(range(1, shape[1] - 1))]

    apply_stencil(
        laplacian2d,
        domain,
        [cartesian_connectivity],
        out,
        [as_field(inp, LocationType.Vertex)],
    )

    ref = np.zeros(shape)
    ref[1:-1, 1:-1] = 2

    assert np.allclose(out, ref)


test_lap()
