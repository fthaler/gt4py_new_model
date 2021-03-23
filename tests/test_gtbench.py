import numpy as np
import pytest

from gt4py_new_model import *


def hdiff_flux(dim):
    @polymorphic_stencil
    def flux(inp, delta):
        f = (
            -inp[dim - 3] * (1.0 / 90)
            + inp[dim - 2] * (5.0 / 36.0)
            - inp[dim - 1] * (49.0 / 36.0)
            + inp[dim] * (49.0 / 36.0)
            - inp[dim + 1] * (5.0 / 36.0)
            + inp[dim + 2] * (1.0 / 90.0)
        ) / delta[()]
        return 0 if f * (inp[()] - inp[dim - 1]) < 0 else f

    return flux


def test_hdiff_flux():
    @fencil
    def apply(inp, delta, out, domain):
        apply_stencil(hdiff_flux(I), domain, [out], [inp, delta])

    rng = np.random.default_rng()
    shape = (2, 3, 2)
    inp = rng.normal(size=(shape[0] + 5, shape[1], shape[2]))
    delta = 0.1
    out = np.zeros(shape)
    inp_s = storage(inp, origin=(3, 0, 0))
    delta_s = constant(delta)
    out_s = storage(out)
    apply(inp_s, delta_s, out_s, domain=domain(shape))

    f = (
        -inp[:-5, :, :] * (1.0 / 90)
        + inp[1:-4, :, :] * (5.0 / 36.0)
        - inp[2:-3, :, :] * (49.0 / 36.0)
        + inp[3:-2, :, :] * (49.0 / 36.0)
        - inp[4:-1, :, :] * (5.0 / 36.0)
        + inp[5:, :, :] * (1.0 / 90.0)
    ) / delta
    flux = np.where(f * (inp[3:-2, :, :] - inp[2:-3, :, :]) < 0, 0, f)

    assert np.allclose(np.asarray(out_s), flux)


@polymorphic_stencil
def hdiff(inp, coeff, dt, dx, dy):
    flx = lift(hdiff_flux(I))(inp, dx)
    fly = lift(hdiff_flux(J))(inp, dy)
    return inp[()] + coeff[()] * dt[()] * (
        (flx[I + 1] - flx[()]) / dx[()] + (fly[J + 1] - fly[()]) / dy[()]
    )


def test_hdiff():
    @fencil
    def apply(inp, coeff, dt, dx, dy, out, domain):
        apply_stencil(hdiff, domain, [out], [inp, coeff, dt, dx, dy])

    rng = np.random.default_rng()
    shape = (2, 3, 2)
    inp = rng.normal(size=(shape[0] + 6, shape[1] + 6, shape[2]))
    coeff = rng.normal(size=shape)
    dx = 0.1
    dy = 0.08
    dt = 0.05
    out = np.zeros(shape)
    inp_s = storage(inp, origin=(3, 3, 0))
    coeff_s = storage(coeff)
    dx_s = constant(dx)
    dy_s = constant(dy)
    dt_s = constant(dt)
    out_s = storage(out)
    apply(inp_s, coeff_s, dt_s, dx_s, dy_s, out_s, domain=domain(shape))

    flx = (
        -inp[:-5, 3:-3, :] * (1.0 / 90)
        + inp[1:-4, 3:-3, :] * (5.0 / 36.0)
        - inp[2:-3, 3:-3, :] * (49.0 / 36.0)
        + inp[3:-2, 3:-3, :] * (49.0 / 36.0)
        - inp[4:-1, 3:-3, :] * (5.0 / 36.0)
        + inp[5:, 3:-3, :] * (1.0 / 90.0)
    ) / dx
    flx = np.where(flx * (inp[3:-2, 3:-3, :] - inp[2:-3, 3:-3, :]) < 0, 0, flx)

    fly = (
        -inp[3:-3, :-5, :] * (1.0 / 90)
        + inp[3:-3, 1:-4, :] * (5.0 / 36.0)
        - inp[3:-3, 2:-3, :] * (49.0 / 36.0)
        + inp[3:-3, 3:-2, :] * (49.0 / 36.0)
        - inp[3:-3, 4:-1, :] * (5.0 / 36.0)
        + inp[3:-3, 5:, :] * (1.0 / 90.0)
    ) / dy
    fly = np.where(fly * (inp[3:-3, 3:-2, :] - inp[3:-3, 2:-3, :]) < 0, 0, fly)

    out = inp[3:-3, 3:-3, :] + coeff * dt * (
        (flx[1:, :, :] - flx[:-1, :, :]) / dx + (fly[:, 1:, :] - fly[:, :-1, :]) / dy
    )

    assert np.allclose(np.asarray(out_s), out)


def hadv_flux(dim, dir):
    @polymorphic_stencil
    def flux(inp, velocity, delta):
        return (
            -velocity[()]
            * (
                inp[dim - dir * 3] * (1.0 / 30.0)
                - inp[dim - dir * 2] * (1.0 / 4.0)
                + inp[dim - dir * 1]
                - inp[dim] * (1.0 / 3.0)
                - inp[dim + dir * 1] * (1.0 / 2.0)
                + inp[dim + dir * 2] * (1.0 / 20.0)
            )
            / delta[()]
        )

    return flux


def hadv_upwind_flux(dim):
    @polymorphic_stencil
    def flux(inp, velocity, delta):
        f_up = lift(hadv_flux(dim, 1))(inp, velocity, delta)
        f_down = lift(hadv_flux(dim, -1))(inp, velocity, delta)
        return f_up if velocity[()] > 0 else f_down

    return flux


@polymorphic_stencil
def hadv(inp, inp0, u, v, dt, dx, dy):
    flx = hadv_upwind_flux(I)(inp, u, dx)
    fly = hadv_upwind_flux(J)(inp, v, dy)
    return inp0[()] - dt[()] * (flx + fly)


def vdiff(k_size):
    k_offset = k_size - 1

    @forward
    def vdiff_forward(state, k, inp, coeff, dt, dz):
        if k[()] == 0:
            a = -coeff[()] / (2 * dz[()] ** 2)
            b = 1 / dt[()] + coeff[()] / dz[()] ** 2
            d1 = (
                1 / dt[()] * inp[()]
                + 0.5
                * coeff[()]
                * (inp[K + k_offset] - 2 * inp[()] + inp[K + 1])
                / dz[()] ** 2
            )
            d2 = -a

            b_k = b
            d1_k = d1
            d2_k = d2
        elif 1 <= k[()] < k_size - 2:
            a = -coeff[()] / (2 * dz[()] ** 2)
            b = 1 / dt[()] + coeff[()] / dz[()] ** 2
            c = a
            d1 = (
                1 / dt[()] * inp[()]
                + 0.5
                * coeff[()]
                * (inp[K - 1] - 2 * inp[()] + inp[K + 1])
                / dz[()] ** 2
            )
            d2 = 0

            b_km1, d1_km1, d2_km1 = state
            f = a / b_km1
            b_k = b - f * c
            d1_k = d1 - f * d1_km1
            d2_k = d2 - f * d2_km1
        elif k[()] == k_size - 2:
            a = -coeff[()] / (2 * dz[()] ** 2)
            b = 1 / dt[()] + coeff[()] / dz[()] ** 2
            c = a
            d1 = (
                1 / dt[()] * inp[()]
                + 0.5
                * coeff[()]
                * (inp[K - 1] - 2 * inp[()] + inp[K + 1])
                / dz[()] ** 2
            )
            d2 = -c

            b_km1, d1_km1, d2_km1 = state
            f = a / b_km1
            b_k = b - f * c
            d1_k = d1 - f * d1_km1
            d2_k = d2 - f * d2_km1
        else:
            b_k = d1_k = d2_k = 0
        return b_k, d1_k, d2_k

    @backward
    def vdiff_backward1(state, k, b, d1, d2, coeff, dz):
        if k[()] == k_size - 1:
            d1_k = d2_k = 0
        else:
            c = -coeff[()] / (2 * dz[()] ** 2)
            f = 1 / b[()]
            d1_kp1, d2_kp1 = state
            d1_k = (d1[()] - c * d1_kp1) * f
            d2_k = (d2[()] - c * d2_kp1) * f
        return d1_k, d2_k

    @backward
    def vdiff_backward2(state, k, d1, d2, inp, coeff, dt, dz):
        if k[()] == k_size - 1:
            a = -coeff[()] / (2 * dz[()] ** 2)
            b = 1 / dt[()] + coeff[()] / dz[()] ** 2
            c = a
            d1_top = (
                1 / dt[()] * inp[()]
                + 0.5
                * coeff[()]
                * (inp[K - 1] - 2 * inp[()] + inp[K - k_offset])
                / dz[()] ** 2
            )

            out = (d1_top - c * d1[K - k_offset] - a * d1[K - 1]) / (
                b + c * d2[K - k_offset] + a * d2[K - 1]
            )
            return out, out
        else:
            _, out_top = state
            out = d1[()] + d2[()] * out_top  # noqa: F841
            return out, out_top

    @stencil
    def vdiff_composed(k, inp, coeff, dt, dz):
        b, d1, d2 = liftv(vdiff_forward)(k, inp, coeff, dt, dz)
        d1p, d2p = liftv(vdiff_backward1)(k, b, d1, d2, coeff, dz)
        out, _ = vdiff_backward2(k, d1p, d2p, inp, coeff, dt, dz)
        return out

    return vdiff_composed


def test_vdiff():
    shape = (2, 3, 5)

    @fencil
    def apply(k, inp, coeff, dt, dz, out, domain):
        apply_stencil(vdiff(shape[2]), domain, [out], [k, inp, coeff, dt, dz])

    rng = np.random.default_rng()
    inp = rng.normal(size=(shape[0] + 6, shape[1] + 6, shape[2]))
    coeff = rng.normal(size=shape)
    dz = 0.1
    dt = 0.05
    out = np.zeros(shape)
    inp_s = storage(inp, origin=(3, 3, 0))
    coeff_s = storage(coeff)
    k_s = index(shape, "k")
    dz_s = constant(dz)
    dt_s = constant(dt)
    out_s = storage(out)
    apply(k_s, inp_s, coeff_s, dt_s, dz_s, out_s, domain=domain(shape))


@polymorphic_stencil
def zero_gradient_bc(
    inp,
    on_lower_i_boundary,
    on_upper_i_boundary,
    on_lower_j_boundary,
    on_upper_j_boundary,
):
    if on_lower_i_boundary[()]:
        return inp[I + 1]
    elif on_upper_i_boundary[()]:
        return inp[I - 1]
    elif on_lower_j_boundary[()]:
        return inp[J + 1]
    elif on_upper_j_boundary[()]:
        return inp[J - 1]
    return inp


def full_diffusion(k_size):
    @stencil
    def diffusion(
        k,
        inp,
        coeff,
        dt,
        dx,
        dy,
        dz,
        on_lower_i_boundary,
        on_upper_i_boundary,
        on_lower_j_boundary,
        on_upper_j_boundary,
    ):
        inp_with_bc = lift(zero_gradient_bc)(
            inp,
            on_lower_i_boundary,
            on_upper_i_boundary,
            on_lower_j_boundary,
            on_upper_j_boundary,
        )
        horizontally_diffused = lift(hdiff)(inp_with_bc, coeff, dt, dx, dy)
        return vdiff(k_size)(k, horizontally_diffused, coeff, dt, dz)


def full_advection(k_size):
    @stencil
    def advection_step(
        k,
        inp,
        inp0,
        u,
        v,
        w,
        dt,
        dx,
        dy,
        dz,
        on_lower_i_boundary,
        on_upper_i_boundary,
        on_lower_j_boundary,
        on_upper_j_boundary,
    ):
        inp_with_bc = lift(zero_gradient_bc)(
            inp,
            on_lower_i_boundary,
            on_upper_i_boundary,
            on_lower_j_boundary,
            on_upper_j_boundary,
        )
        horizontally_advected = lift(hadv)(inp_with_bc, inp_with_bc, u, v, dt, dx, dy)
        return vadv(k_size)(k, inp_with_bc, horizontally_advected, w, dt, dz)

    @stencil
    def rk_advection(
        k,
        inp,
        u,
        v,
        w,
        dt,
        dx,
        dy,
        dz,
        on_lower_i_boundary,
        on_upper_i_boundary,
        on_lower_j_boundary,
        on_upper_j_boundary,
    ):
        out1 = lift(advection_step)(
            k,
            inp,
            inp,
            u,
            v,
            w,
            dt / 3,
            dx,
            dy,
            dz,
            on_lower_i_boundary,
            on_upper_i_boundary,
            on_lower_j_boundary,
            on_upper_j_boundary,
        )
        out2 = lift(advection_step)(
            k,
            out1,
            inp,
            u,
            v,
            w,
            dt / 2,
            dx,
            dy,
            dz,
            on_lower_i_boundary,
            on_upper_i_boundary,
            on_lower_j_boundary,
            on_upper_j_boundary,
        )
        return advection_step(
            k,
            out2,
            inp,
            u,
            v,
            w,
            dt,
            dx,
            dy,
            dz,
            on_lower_i_boundary,
            on_upper_i_boundary,
            on_lower_j_boundary,
            on_upper_j_boundary,
        )

    return rk_advection


def gtbench_step(k_size):
    @stencil
    def step(
        k,
        inp,
        diff_coeff,
        u,
        v,
        w,
        dt,
        dx,
        dy,
        dz,
        on_lower_i_boundary,
        on_upper_i_boundary,
        on_lower_j_boundary,
        on_upper_j_boundary,
    ):
        inp_with_bc = lift(zero_gradient_bc)(
            inp,
            on_lower_i_boundary,
            on_upper_i_boundary,
            on_lower_j_boundary,
            on_upper_j_boundary,
        )
        horizontally_diffused = lift(hdiff)(inp_with_bc, diff_coeff, dt, dx, dy)
        advected = lift(full_advection(k_size))(
            k,
            horizontally_diffused,
            u,
            v,
            w,
            dt,
            dx,
            dy,
            dz,
            on_lower_i_boundary,
            on_upper_i_boundary,
            on_lower_j_boundary,
            on_upper_j_boundary,
        )
        return vdiff(k_size)(k, advected, diff_coeff, dt, dz)
