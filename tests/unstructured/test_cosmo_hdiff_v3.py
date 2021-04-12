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
from unstructured.cartesian import CartesianNeighborHood, cartesian_connectivity


cart = CartesianNeighborHood()


@stencil((cart,))
def laplacian(inp):
    return -4 * inp[0, 0] + (inp[-1, 0] + inp[1, 0] + inp[0, -1] + inp[0, 1])


@stencil((cart,))
def hdiff_flux_x(inp):
    lap = lift(laplacian)(inp)
    flux = lap[0, 0] - lap[1, 0]

    return 0 if flux * (inp[1, 0] - inp[0, 0]) > 0 else flux


@stencil((cart,))
def hdiff_flux_y(inp):
    lap = lift(laplacian)(inp)
    flux = lap[0, 0] - lap[0, 1]

    return 0 if flux * (inp[0, 1] - inp[0, 0]) > 0 else flux


@stencil((cart,), ())
def hdiff(inp, coeff):
    flx = lift(hdiff_flux_x)(inp)
    fly = lift(hdiff_flux_y)(inp)
    return inp[0, 0] - coeff * (flx[0, 0] - flx[-1, 0] + fly[0, 0] - fly[0, -1])


@pytest.fixture
def hdiff_reference():
    shape = (5, 7, 5)
    rng = np.random.default_rng()
    inp = rng.normal(size=(shape[0] + 4, shape[1] + 4, shape[2]))
    coeff = rng.normal(size=shape)

    lap = 4 * inp[1:-1, 1:-1, :] - (
        inp[2:, 1:-1, :] + inp[:-2, 1:-1, :] + inp[1:-1, 2:, :] + inp[1:-1, :-2, :]
    )
    uflx = lap[1:, 1:-1, :] - lap[:-1, 1:-1, :]
    flx = np.where(uflx * (inp[2:-1, 2:-2, :] - inp[1:-2, 2:-2, :]) > 0, 0, uflx)
    ufly = lap[1:-1, 1:, :] - lap[1:-1, :-1, :]
    fly = np.where(ufly * (inp[2:-2, 2:-1, :] - inp[2:-2, 1:-2, :]) > 0, 0, ufly)
    out = inp[2:-2, 2:-2, :] - coeff * (
        flx[1:, :, :] - flx[:-1, :, :] + fly[:, 1:, :] - fly[:, :-1, :]
    )

    return inp, coeff, out


def test_hdiff(hdiff_reference):
    inp, coeff, out = hdiff_reference
    shape = (inp.shape[0], inp.shape[1])
    inp_s = as_field(inp[:, :, 0], LocationType.Vertex)
    coeff_full_domain = np.zeros(shape)
    coeff_full_domain[2:-2, 2:-2] = coeff[:, :, 0]
    coeff_s = as_field(coeff_full_domain, LocationType.Vertex)
    out_s = np.zeros_like(inp)[:, :, 0]

    inner_domain = [list(range(2, shape[0] - 2)), list(range(2, shape[1] - 2))]

    # TODO: lift needs to deal with the folded accessors (i.e. cartesian_acc(cartesian_acc(...)) is folded to cartesian_acc(...))
    # apply_stencil(
    #     hdiff,
    #     inner_domain,
    #     [cartesian_connectivity],
    #     out_s,
    #     [inp_s, coeff_s],
    # )

    # assert np.allclose(out[:, :, 0], out_s[2:-2, 2:-2])


# test_hdiff(hdiff_reference())
