import math
import numpy as np
import pytest

from unstructured.concepts import (
    LocationType,
    apply_stencil,
    connectivity,
    lift,
    neighborhood,
    stencil,
    ufield,
)
from unstructured.helpers import as_1d, as_2d, as_field


@neighborhood(LocationType.Vertex, LocationType.Vertex)
class FivePointNeighborHood:
    center = 0
    left = 1
    right = 2
    top = 3
    bottom = 4


fp = FivePointNeighborHood()


@stencil((fp,))
def laplacian(inp):
    return -4 * inp[fp.center] + (
        inp[fp.right] + inp[fp.left] + inp[fp.bottom] + inp[fp.top]
    )


@stencil((fp,), (fp, fp))
def hdiff_flux_x(inp1, inp2):
    lap = lift(laplacian)(inp2)
    flux = lap[fp.center] - lap[fp.right]

    return 0 if flux * (inp1[fp.right] - inp1[fp.center]) > 0 else flux


@stencil((fp,), (fp, fp))
def hdiff_flux_y(inp1, inp2):
    lap = lift(laplacian)(inp2)
    flux = lap[fp.center] - lap[fp.bottom]

    return 0 if flux * (inp1[fp.bottom] - inp1[fp.center]) > 0 else flux


@stencil((fp,), (fp, fp), (fp, fp, fp), ())
def hdiff(inp1, inp2, inp3, coeff):
    flx = lift(hdiff_flux_x)(inp2, inp3)
    fly = lift(hdiff_flux_y)(inp2, inp3)
    return inp1[fp.center] - coeff * (
        flx[fp.center] - flx[fp.left] + fly[fp.center] - fly[fp.top]
    )


# @pytest.fixture
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


def make_fpconn(shape):
    strides = [shape[1], 1]

    @connectivity(fp)
    def v2v_conn(field):
        @ufield(LocationType.Vertex)
        def _field(index):
            return [
                field(index),
                field(index - strides[0]),
                field(index + strides[0]),
                field(index - strides[1]),
                field(index + strides[1]),
            ]

        return _field

    return v2v_conn


def test_hdiff(hdiff_reference):

    inp, coeff, out = hdiff_reference
    shape = (inp.shape[0], inp.shape[1])
    inp_s = as_field(as_1d(inp[:, :, 0]), LocationType.Vertex)
    coeff_full_domain = np.zeros(shape)  # _like(inp)
    coeff_full_domain[2:-2, 2:-2] = coeff[:, :, 0]
    coeff_s = as_field(as_1d(coeff_full_domain), LocationType.Vertex)
    out_s = as_1d(np.zeros_like(inp)[:, :, 0])

    domain = np.arange(math.prod(shape))
    domain_2d = as_2d(domain, shape)
    # inner_domain = as_1d(domain_2d[2:-2, 2:-2])
    inner_domain = as_1d(domain_2d[3:-3, 3:-3])

    apply_stencil(
        hdiff, inner_domain, [make_fpconn(shape)], out_s, [inp_s, inp_s, inp_s, coeff_s]
    )

    assert np.allclose(out[1:-1, 1:-1, 0], np.asarray(as_2d(out_s, shape)[3:-3, 3:-3]))


test_hdiff(hdiff_reference())
