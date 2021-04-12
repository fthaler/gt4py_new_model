import numpy as np
from numpy.core.numeric import outer

from unstructured.concepts import (
    LocationType,
    apply_stencil,
    connectivity,
    neighborhood,
    stencil,
    ufield,
)
from unstructured.helpers import as_field, simple_connectivity
from unstructured.cartesian import CartesianNeighborHood

cart = CartesianNeighborHood()


def cartesian_accessor(field, *indices):
    class _cartesian_accessor:
        def __call__(self):
            return field(*indices)

        def __getitem__(self, neighindices):
            if not isinstance(neighindices, tuple):
                neighindices = (neighindices,)
            return cartesian_accessor(
                field,
                *tuple(
                    map(lambda x: x[0] + x[1], zip(indices, neighindices)),
                )
            )

    return _cartesian_accessor()


@connectivity(cart)
def cartesian_connectivity2(field):
    @ufield(cart.in_location)
    def _field(*index):

        return cartesian_accessor(field, *index)

    return _field


def test_cartesian_connectivity():
    inp = np.arange(10 * 10).reshape(10, 10)
    print(inp)

    inp_s = as_field(inp, LocationType.Vertex)

    assert cartesian_accessor(inp_s, 1, 1)[1, 1][2, 2]() == 44
    acc_field = cartesian_connectivity2(inp_s)

    assert inp_s(1, 1) == 11
    assert acc_field(1, 1)[1, 1]() == 22
    assert acc_field(1, 1)[1, 1][2, 2]() == 44


test_cartesian_connectivity()
