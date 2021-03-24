import numpy as np
import pytest

from gt4py_new_model import *


@polymorphic_stencil
def laplacian(inp):
    return -4 * inp[()] + (inp[I + 1] + inp[I - 1] + inp[J + 1] + inp[J - 1])


def hdiff_flux(dim):
    @polymorphic_stencil
    def res(inp):
        lap = lift(laplacian)(inp)
        uflux = lap[()] - lap[dim + 1]
        return 0 if uflux * (inp[dim + 1] - inp[()]) > 0 else uflux

    return res


@polymorphic_stencil
def hdiff(inp, coeff):
    flx = lift(hdiff_flux(I))(inp)
    fly = lift(hdiff_flux(J))(inp)
    return inp[()] - coeff[()] * (flx[()] - flx[I - 1] + fly[()] - fly[J - 1])


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
    @fencil
    def apply(inp, coeff, out, domain):
        apply_stencil(hdiff, domain, [out], [inp, coeff])

    inp, coeff, out = hdiff_reference
    inp_s = storage(inp, origin=(2, 2, 0))
    coeff_s = storage(coeff)
    out_s = storage(np.zeros_like(out))
    apply(inp_s, coeff_s, out_s, domain=domain(out.shape))
    assert np.allclose(out, np.asarray(out_s))


@pytest.fixture
def tridiag_reference():
    shape = (3, 7, 5)
    rng = np.random.default_rng()
    a = rng.normal(size=shape)
    b = rng.normal(size=shape) * 2
    c = rng.normal(size=shape)
    d = rng.normal(size=shape)

    matrices = np.zeros(shape + shape[-1:])
    i = np.arange(shape[2])
    matrices[:, :, i[1:], i[:-1]] = a[:, :, 1:]
    matrices[:, :, i, i] = b
    matrices[:, :, i[:-1], i[1:]] = c[:, :, :-1]
    x = np.linalg.solve(matrices, d)
    return a, b, c, d, x


@forward
def tridiag_forward(state, a, b, c, d):
    if state is None:
        cp_k = c[K] / b[K]
        dp_k = d[K] / b[K]
    else:
        cp_km1, dp_km1 = state
        cp_k = c[K] / (b[K] - a[K] * cp_km1)
        dp_k = (d[K] - a[K] * dp_km1) / (b[K] - a[K] * cp_km1)
    return cp_k, dp_k


@backward
def tridiag_backward(x_kp1, cp, dp):
    if x_kp1 is None:
        x_k = dp[K]
    else:
        x_k = dp[K] - cp[K] * x_kp1
    return x_k


@stencil
def solve_tridiag(a, b, c, d):
    cp, dp = lift(tridiag_forward)(a, b, c, d)
    return tridiag_backward(cp, dp)


def test_tridiag(tridiag_reference):
    @fencil
    def apply(x, a, b, c, d, domain):
        apply_stencil(solve_tridiag, domain, [x], [a, b, c, d])

    a, b, c, d, x = tridiag_reference
    a_s = storage(a)
    b_s = storage(b)
    c_s = storage(c)
    d_s = storage(d)
    x_s = storage(np.zeros_like(x))
    apply(x_s, a_s, b_s, c_s, d_s, domain=domain(x.shape))

    assert np.allclose(x, np.asarray(x_s))


def test_combined():
    @stencil
    def hdiff_tridiag(a, b, c, d, coeff):
        x = lift(solve_tridiag)(a, b, c, d)
        return hdiff(x, coeff)

    @fencil
    def apply(out, a, b, c, d, coeff, domain):
        apply_stencil(hdiff_tridiag, domain, [out], [a, b, c, d, coeff])

    rng = np.random.default_rng()
    shape = (2, 1, 3)
    a = rng.normal(size=(shape[0] + 4, shape[1] + 4, shape[2]))
    b = rng.normal(size=a.shape)
    c = rng.normal(size=a.shape)
    d = rng.normal(size=a.shape)
    coeff = rng.normal(size=shape)
    a_s = storage(a, origin=(2, 2, 0))
    b_s = storage(b, origin=(2, 2, 0))
    c_s = storage(c, origin=(2, 2, 0))
    d_s = storage(d, origin=(2, 2, 0))
    coeff_s = storage(coeff)
    out_s = storage(np.zeros(shape))

    apply(out_s, a_s, b_s, c_s, d_s, coeff_s, domain=domain(shape))
