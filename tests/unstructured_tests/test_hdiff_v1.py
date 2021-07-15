from unstructured.builtins import *
from unstructured.runtime import *
from unstructured.embedded import np_as_located_field
import numpy as np
from .hdiff_reference import hdiff_reference

I = offset("I")
J = offset("J")

IDim = CartesianAxis("IDim")
JDim = CartesianAxis("JDim")


@fundef
def laplacian(inp):
    return -4.0 * deref(inp) + (
        deref(shift(I, 1)(inp))
        + deref(shift(I, -1)(inp))
        + deref(shift(J, 1)(inp))
        + deref(shift(J, -1)(inp))
    )


@fundef
def flux(d):
    def flux_impl(inp):
        lap = lift(laplacian)(inp)
        flux = deref(lap) - deref(shift(d, 1)(lap))
        return if_(flux * (deref(shift(d, 1)(inp)) - deref(inp)) > 0.0, 0.0, flux)

    return flux_impl


@fundef
def flux_I(inp):
    return flux(I)(inp)


@fundef
def hdiff_sten(inp, coeff):
    flx = lift(flux(I))(inp)
    fly = lift(flux(J))(inp)
    return deref(inp) - (
        deref(coeff)
        * (
            deref(flx)
            - deref(shift(I, -1)(flx))
            + deref(fly)
            - deref(shift(J, -1)(fly))
        )
    )


@fendef(offset_provider={"I": IDim, "J": JDim})
def hdiff(inp, coeff, out, x, y):
    closure(
        domain(named_range(IDim, 0, x), named_range(JDim, 0, y)),
        hdiff_sten,
        [out],
        [inp, coeff],
    )


hdiff(*([None] * 5), backend="lisp")
hdiff(*([None] * 5), backend="cpptoy")


def test_hdiff(hdiff_reference):
    inp, coeff, out = hdiff_reference
    shape = (out.shape[0], out.shape[1])

    inp_s = np_as_located_field(IDim, JDim, origin={IDim: 2, JDim: 2})(inp[:, :, 0])
    coeff_s = np_as_located_field(IDim, JDim)(coeff[:, :, 0])
    out_s = np_as_located_field(IDim, JDim)(np.zeros_like(coeff[:, :, 0]))

    # hdiff(inp_s, coeff_s, out_s, shape[0], shape[1])
    # hdiff(inp_s, coeff_s, out_s, shape[0], shape[1], backend="embedded")
    hdiff(inp_s, coeff_s, out_s, shape[0], shape[1], backend="double_roundtrip")

    assert np.allclose(out[:, :, 0], out_s)
