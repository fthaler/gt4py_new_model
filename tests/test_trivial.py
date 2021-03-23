import numpy as np

from typing import Any

from gt4py_new_model import *


def test_identity():
    @polymorphic_stencil
    def identity(inp):
        return inp[I, J, K]

    @fencil
    def apply(out, inp, domain):
        apply_stencil(identity, domain, [out], [inp])

    shape = (3, 5, 7)
    rng = np.random.default_rng()
    inp = storage(rng.normal(size=shape))
    out = storage(rng.normal(size=shape))
    apply(out, inp, domain=domain(shape))
    assert np.all(np.asarray(inp) == np.asarray(out))


@polymorphic_stencil
def shift(inp):
    return inp[I + 1]


def test_shift():
    @fencil
    def apply(out, inp, domain):
        apply_stencil(shift, domain, [out], [inp])

    shape = (3, 5, 7)
    rng = np.random.default_rng()
    inp = storage(rng.normal(size=shape))
    out = storage(rng.normal(size=shape))
    apply(out, inp, domain=domain(shape[0] - 1, shape[1], shape[2]))
    assert np.all(np.asarray(inp)[1:, :, :] == np.asarray(out)[:-1, :, :])


@polymorphic_stencil
def scale(inp):
    return 2 * inp[I]


def test_scale():
    @fencil
    def apply(out, inp, domain):
        apply_stencil(scale, domain, [out], [inp])

    shape = (3, 5, 7)
    rng = np.random.default_rng()
    inp = storage(rng.normal(size=shape))
    out = storage(rng.normal(size=shape))
    apply(out, inp, domain=domain(shape))
    assert np.allclose(2 * np.asarray(inp), np.asarray(out))


def test_shift_scale():
    @polymorphic_stencil
    def shift_scale(inp):
        x = lift(shift)(inp)
        return scale(x)

    @fencil
    def apply(out, inp, domain):
        apply_stencil(shift_scale, domain, [out], [inp])

    shape = (3, 5, 7)
    rng = np.random.default_rng()
    inp = storage(rng.normal(size=shape))
    out = storage(rng.normal(size=shape))
    apply(out, inp, domain=domain(shape[0] - 1, shape[1], shape[2]))
    assert np.allclose(2 * np.asarray(inp)[1:, :, :], np.asarray(out)[:-1, :, :])


def test_multi_shift_scale():
    @polymorphic_stencil
    def shift_scale(inp):
        x = lift(shift)(inp)
        y = lift(scale)(x)
        z = lift(lambda x, y: shift(y) * x[I])(x, y)
        return scale(z)

    @fencil
    def apply(out, inp, domain):
        apply_stencil(shift_scale, domain, [out], [inp])

    shape = (3, 5, 7)
    rng = np.random.default_rng()
    inp = storage(rng.normal(size=shape))
    out = storage(rng.normal(size=shape))
    apply(out, inp, domain=domain(shape[0] - 2, shape[1], shape[2]))
    x = np.asarray(inp)[1:, :, :]
    y = 2 * x
    z = y[1:, :, :] * x[:-1, :, :]
    res = 2 * z
    assert np.allclose(res, np.asarray(out)[:-2, :, :])


def test_cumsum():
    @forward(init=0)
    def cumsum(state, inp):
        return state + inp[K]

    @fencil
    def apply(out, inp, domain):
        apply_stencil(cumsum, domain, [out], [inp])

    shape = (3, 2, 7)
    rng = np.random.default_rng()
    inp = storage(rng.normal(size=shape))
    out = storage(rng.normal(size=shape))
    apply(out, inp, domain=domain(shape))
    assert np.all(np.cumsum(inp, axis=2) == np.asarray(out))


def test_conditional():
    @polymorphic_stencil
    def foo(x, y, z):
        return y[I] if x[I] else z[I]

    @fencil
    def apply(out, x, y, z, domain):
        apply_stencil(foo, domain, [out], [x, y, z])

    shape = (3, 2, 7)
    rng = np.random.default_rng()
    x = storage(rng.normal(size=shape) > 0)
    y = storage(rng.normal(size=shape))
    z = storage(rng.normal(size=shape))
    out = storage(rng.normal(size=shape))
    apply(out, x, y, z, domain=domain(shape))
    assert np.all(
        np.asarray(out) == np.where(np.asarray(x), np.asarray(y), np.asarray(z))
    )


def test_tuple_lifting():
    @polymorphic_stencil
    def foo(x, y):
        return x[I], y[I]

    @polymorphic_stencil
    def bar(x, y):
        xx, yy = lift(foo)(x, y)
        return xx[I] + yy[I]

    @fencil
    def apply(out, x, y, domain):
        apply_stencil(bar, domain, [out], [x, y])

    shape = (3, 2, 7)
    rng = np.random.default_rng()
    x = storage(rng.normal(size=shape))
    y = storage(rng.normal(size=shape))
    out = storage(rng.normal(size=shape))
    apply(out, x, y, domain=domain(shape))
    assert np.allclose(np.asarray(out), np.asarray(x) + np.asarray(y))
