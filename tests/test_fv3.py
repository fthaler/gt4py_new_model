# GT4Py New Semantic Model - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.  GT4Py
# New Semantic Model is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or any later version.
# See the LICENSE.txt file at the top-level directory of this distribution for
# a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

from gt4py_new_model import *


def ifs1_reference():
    rng = np.random.default_rng()
    shape = (3, 5, 7)
    qi_in = rng.normal(size=shape)
    qs_in = rng.normal(size=shape)
    qg_in = rng.normal(size=shape)
    qi_out = np.copy(qi_in)
    qs_out = np.copy(qs_in)
    qg_out = np.copy(qg_in)
    dq_tmp = np.empty_like(qi_in)

    with np.nditer(
        [qi_out, qs_out, qg_out, dq_tmp], op_flags=[["readwrite"]] * 4
    ) as it:

        for qi, qs, qg, dq in it:
            # https://github.com/GridTools/concepts/issues/22#issuecomment-686556399
            qsum = qi + qs
            if qsum > 0.0:
                if qi < 0.0:
                    qi[...] = 0.0
                    qs[...] = qsum
                elif qs < 0.0:
                    qs[...] = 0.0
                    qi[...] = qsum
            else:
                qi[...] = 0.0
                qs[...] = 0.0
                qg[...] = qg + qsum
            if qg < 0.0:
                dq[...] = qs if qs < -qg else -qg
                qs[...] = qs - dq
                qg[...] = qg + dq
                if qg < 0.0:
                    dq[...] = qi if qi < -qg else -qg
                    qi[...] = qi - dq
                    qg[...] = qg + dq

    return (qi_in, qs_in, qg_in), (qi_out, qs_out, qg_out)


@polymorphic_stencil
def ifs1_impl1(qi, qs, qg):
    qsum = qi[()] + qs[()]
    if qsum > 0.0:
        if qi[()] < 0.0:
            qi1 = 0.0
            qs1 = qsum
        elif qs[()] < 0.0:
            qi1 = qsum
            qs1 = 0.0
        else:
            qi1 = qi[()]
            qs1 = qs[()]
        qg1 = qg[()]
    else:
        qi1 = 0.0
        qs1 = 0.0
        qg1 = qg[()] + qsum
    if qg1 < 0.0:
        dq = qs1 if qs1 < -qg1 else -qg1
        qs2 = qs1 - dq
        qg2 = qg1 + dq
        if qg2 < 0.0:
            dq1 = qi1 if qi1 < -qg2 else -qg2
            qi2 = qi1 - dq1
            qg3 = qg2 + dq1
        else:
            qi2 = qi1
            qg3 = qg2
    else:
        qs2 = qs1
        qi2 = qi1
        qg3 = qg1
    return qi2, qs2, qg3


def ifs1_fun1(qi, qs, qg):
    qsum = qi + qs
    return (
        (0.0, 0.0, qg + qsum)
        if qsum <= 0.0
        else (0.0, qsum, qg)
        if qi < 0.0
        else (qsum, 0.0, qg)
        if qs < 0.0
        else (qi, qs, qg)
    )


def ifs1_fun2(qx, qg):
    dq = qx if qx < -qg else -qg
    return qx - dq, qg + dq


def ifs1_fun3(qi, qs, qg):
    qs1, qg1 = ifs1_fun2(qs, qg)
    qi1, qg2 = ifs1_fun2(qi, qg1)
    return (
        (qi, qs, qg) if qg >= 0.0 else (qi, qs1, qg1) if qg1 >= 0.0 else (qi1, qs1, qg2)
    )


@polymorphic_stencil
def ifs1_impl2(qi, qs, qg):
    qi1, qs1, qg1 = ifs1_fun1(qi[()], qs[()], qg[()])
    return ifs1_fun3(qi1, qs1, qg1)


def test_ifs1():
    (qi_in, qs_in, qg_in), (qi_out, qs_out, qg_out) = ifs1_reference()

    @fencil
    def apply(qi_in, qs_in, qg_in, qi_out, qs_out, qg_out, domain):
        apply_stencil(
            ifs1_impl2, domain, [qi_out, qs_out, qg_out], [qi_in, qs_in, qg_in]
        )

    qi_in_s = storage(qi_in)
    qs_in_s = storage(qs_in)
    qg_in_s = storage(qg_in)
    qi_out_s = storage(np.zeros_like(qi_in))
    qs_out_s = storage(np.zeros_like(qs_in))
    qg_out_s = storage(np.zeros_like(qg_in))

    apply(
        qi_in_s,
        qs_in_s,
        qg_in_s,
        qi_out_s,
        qs_out_s,
        qg_out_s,
        domain=domain(qi_in.shape),
    )

    assert np.allclose(qi_out, qi_out_s)
    assert np.allclose(qs_out, qs_out_s)
    assert np.allclose(qg_out, qg_out_s)


def no_idea_what_it_is_doing_reference():
    rng = np.random.default_rng()
    shape = (3, 5, 7)
    qv_in = rng.normal(size=shape)
    dp = rng.normal(size=shape)
    qv = np.copy(qv_in)

    for k in range(1, shape[2] - 1):
        # https://github.com/GridTools/concepts/issues/22#issuecomment-686555219
        dq = np.minimum(-qv[:, :, k] * dp[:, :, k], qv[:, :, k - 1] * dp[:, :, k - 1])
        mask = (qv[:, :, k] < 0) & (qv[:, :, k - 1] > 0.0)
        qv[:, :, k - 1][mask] -= (dq / dp[:, :, k - 1])[mask]
        qv[:, :, k][mask] += (dq / dp[:, :, k])[mask]
        mask = qv[:, :, k] < 0.0
        qv[:, :, k + 1][mask] += (qv[:, :, k] * dp[:, :, k] / dp[:, :, k + 1])[mask]
        qv[:, :, k][mask] = 0.0

    return (qv_in, dp), qv


def do_what_you_want_with_your_column(func):
    def wrapper(*args):
        columns = (np.copy(arg[()]) for arg in args)
        return func(*columns)

    return wrapper


@do_what_you_want_with_your_column
def no_idea_what_it_is_doing_impl(qv, dp):
    for k in range(1, qv.size - 1):
        if qv[k] < 0 and qv[k - 1] > 0.0:
            dq = min(-qv[k] * dp[k], qv[k - 1] * dp[k - 1])
            qv[k - 1] -= dq / dp[k - 1]
            qv[k] += dq / dp[k]
        if qv[k] < 0.0:
            qv[k + 1] += qv[k] * dp[k] / dp[k + 1]
            qv[k] = 0.0
    return qv


def test_no_idea_what_it_is_doing():
    (qv_in, dp), qv_out = no_idea_what_it_is_doing_reference()

    @fencil
    def apply(qv_in, dp, qv_out, domain):
        apply_stencil(no_idea_what_it_is_doing_impl, domain, [qv_out], [qv_in, dp])

    qv_in_s = storage(qv_in)
    dp_s = storage(dp)
    qv_out_s = storage(np.zeros_like(qv_out))
    apply(qv_in_s, dp_s, qv_out_s, domain=domain(qv_in.shape))

    assert np.allclose(qv_out, qv_out_s)


def divergence_corner_reference():
    shape = (5, 7, 3)
    rng = np.random.default_rng()
    u = rng.normal(size=shape)
    v = rng.normal(size=shape)
    ua = rng.normal(size=shape)
    va = rng.normal(size=shape)
    dxc = rng.normal(size=shape[:2])
    dyc = rng.normal(size=shape[:2])
    sin_sg1 = rng.normal(size=shape[:2])
    sin_sg2 = rng.normal(size=shape[:2])
    sin_sg3 = rng.normal(size=shape[:2])
    sin_sg4 = rng.normal(size=shape[:2])
    cos_sg1 = rng.normal(size=shape[:2])
    cos_sg2 = rng.normal(size=shape[:2])
    cos_sg3 = rng.normal(size=shape[:2])
    cos_sg4 = rng.normal(size=shape[:2])
    rarea_c = rng.normal(size=shape[:2])
    divg_d = rng.normal(size=shape)
    uf = np.zeros_like(u)
    vf = np.zeros_like(v)

    i_start = 1
    i_end = shape[0] - 2
    j_start = 1
    j_end = shape[1] - 2

    uf[:, 1:, :] = (
        (
            u[:, 1:, :]
            - 0.25
            * (va[:, :-1, :] + va[:, 1:, :])
            * (cos_sg4[:, :-1, np.newaxis] + cos_sg2[:, 1:, np.newaxis])
        )
        * dyc[:, 1:, np.newaxis]
        * 0.5
        * (sin_sg4[:, :-1, np.newaxis] + sin_sg2[:, 1:, np.newaxis])
    )
    uf[:, j_start, :] = (
        u[:, j_start, :]
        * dyc[:, j_start, np.newaxis]
        * 0.5
        * (sin_sg4[:, j_start - 1, np.newaxis] + sin_sg2[:, j_start, np.newaxis])
    )
    uf[:, j_end + 1] = (
        u[:, j_end + 1, :]
        * dyc[:, j_end + 1, np.newaxis]
        * 0.5
        * (sin_sg4[:, j_end, np.newaxis] + sin_sg2[:, j_end + 1, np.newaxis])
    )

    vf[1:, :, :] = (
        (
            v[1:, :, :]
            - 0.25
            * (ua[:-1, :, :] + ua[1:, :, :])
            * (cos_sg3[:-1, :, np.newaxis] + cos_sg1[1:, :, np.newaxis])
        )
        * dxc[1:, :, np.newaxis]
        * 0.5
        * (sin_sg3[:-1, :, np.newaxis] + sin_sg1[1:, :, np.newaxis])
    )
    vf[i_start, :, :] = (
        v[i_start, :, :]
        * dxc[i_start, :, np.newaxis]
        * 0.5
        * (sin_sg3[i_start - 1, :, np.newaxis] + sin_sg1[i_start, :, np.newaxis])
    )
    vf[i_end + 1, :, :] = (
        v[i_end + 1, :, :]
        * dxc[i_end + 1, :, np.newaxis]
        * 0.5
        * (sin_sg3[i_end, :, np.newaxis] + sin_sg1[i_end + 1, :, np.newaxis])
    )

    divg_d[1:, 1:, :] = vf[1:, :-1, :] - vf[1:, 1:, :] + uf[:-1, 1:, :] - uf[1:, 1:, :]
    divg_d[i_start, j_start] -= vf[i_start, j_start - 1, :]
    divg_d[i_end + 1, j_start] -= vf[i_end + 1, j_start - 1, :]
    divg_d[i_end + 1, j_end + 1, :] += vf[i_end + 1, j_end + 1, :]
    divg_d[i_start, j_end + 1, :] += vf[i_start, j_end + 1, :]

    divg_d *= rarea_c[:, :, np.newaxis]

    return (
        i_start,
        i_end,
        j_start,
        j_end,
        u,
        v,
        ua,
        va,
        dxc,
        dyc,
        sin_sg1,
        sin_sg2,
        sin_sg3,
        sin_sg4,
        cos_sg1,
        cos_sg2,
        cos_sg3,
        cos_sg4,
        rarea_c,
    ), divg_d[i_start : i_end + 2, j_start : j_end + 2]


def divergence_corner_impl(i_start, i_end, j_start, j_end):
    @polymorphic_stencil
    def ufs(j, u, va, dyc, sin_sg2, sin_sg4, cos_sg2, cos_sg4):
        if j[()] == j_start or j[()] == j_end + 1:
            return u[()] * dyc[()] * 0.5 * (sin_sg4[J - 1] + sin_sg2[()])
        else:
            return (
                (u[()] - 0.25 * (va[J - 1] + va[()]) * (cos_sg4[J - 1] + cos_sg2[()]))
                * dyc[()]
                * 0.5
                * (sin_sg4[J - 1] + sin_sg2[()])
            )

    @polymorphic_stencil
    def vfs(i, v, ua, dxc, sin_sg1, sin_sg3, cos_sg1, cos_sg3):
        if i[()] == i_start or i[()] == i_end + 1:
            return v[()] * dxc[()] * 0.5 * (sin_sg3[I - 1] + sin_sg1[()])
        else:
            return (
                (v[()] - 0.25 * (ua[I - 1] + ua[()]) * (cos_sg3[I - 1] + cos_sg1[()]))
                * dxc[()]
                * 0.5
                * (sin_sg3[I - 1] + sin_sg1[()])
            )

    @polymorphic_stencil
    def divgs(i, j, uf, vf, rarea_c):
        divg = vf[J - 1] - vf[()] + uf[I - 1] - uf[()]
        ij = (i[()], j[()])
        j_start_corners = ((i_start, j_start), (i_end + 1, j_start))
        i_start_corners = ((i_end + 1, j_end + 1), (i_start, j_end + 1))
        if ij in j_start_corners:
            return (divg - vf[J - 1]) * rarea_c[()]
        elif ij in i_start_corners:
            return (divg + vf[()]) * rarea_c[()]
        else:
            return divg * rarea_c[()]

    @polymorphic_stencil
    def impl(
        i,
        j,
        u,
        v,
        ua,
        va,
        dxc,
        dyc,
        sin_sg1,
        sin_sg2,
        sin_sg3,
        sin_sg4,
        cos_sg1,
        cos_sg2,
        cos_sg3,
        cos_sg4,
        rarea_c,
    ):
        uf = lift(ufs)(j, u, va, dyc, sin_sg2, sin_sg4, cos_sg2, cos_sg4)
        vf = lift(vfs)(i, v, ua, dxc, sin_sg1, sin_sg3, cos_sg1, cos_sg3)
        return divgs(i, j, uf, vf, rarea_c)

    return impl


def test_divergence_corner():
    (
        i_start,
        i_end,
        j_start,
        j_end,
        u,
        v,
        ua,
        va,
        dxc,
        dyc,
        sin_sg1,
        sin_sg2,
        sin_sg3,
        sin_sg4,
        cos_sg1,
        cos_sg2,
        cos_sg3,
        cos_sg4,
        rarea_c,
    ), divg_d = divergence_corner_reference()
    u_s = storage(u, origin=(1, 1, 0))
    v_s = storage(v, origin=(1, 1, 0))
    ua_s = storage(ua, origin=(1, 1, 0))
    va_s = storage(va, origin=(1, 1, 0))
    dxc_s = storage(dxc, origin=(1, 1))
    dyc_s = storage(dyc, origin=(1, 1))
    sin_sg1_s = storage(sin_sg1, origin=(1, 1))
    sin_sg2_s = storage(sin_sg2, origin=(1, 1))
    sin_sg3_s = storage(sin_sg3, origin=(1, 1))
    sin_sg4_s = storage(sin_sg4, origin=(1, 1))
    cos_sg1_s = storage(cos_sg1, origin=(1, 1))
    cos_sg2_s = storage(cos_sg2, origin=(1, 1))
    cos_sg3_s = storage(cos_sg3, origin=(1, 1))
    cos_sg4_s = storage(cos_sg4, origin=(1, 1))
    rarea_c_s = storage(rarea_c, origin=(1, 1))
    divg_d_s = storage(np.zeros_like(divg_d))
    i_s = index(u.shape, "i", origin=(1, 1, 0))
    j_s = index(u.shape, "j", origin=(1, 1, 0))

    @fencil
    def apply(
        i,
        j,
        u,
        v,
        ua,
        va,
        dxc,
        dyc,
        sin_sg1,
        sin_sg2,
        sin_sg3,
        sin_sg4,
        cos_sg1,
        cos_sg2,
        cos_sg3,
        cos_sg4,
        rarea_c,
        divg_d,
        domain,
    ):
        apply_stencil(
            divergence_corner_impl(i_start, i_end, j_start, j_end),
            domain,
            [divg_d],
            [
                i,
                j,
                u,
                v,
                ua,
                va,
                dxc,
                dyc,
                sin_sg1,
                sin_sg2,
                sin_sg3,
                sin_sg4,
                cos_sg1,
                cos_sg2,
                cos_sg3,
                cos_sg4,
                rarea_c,
            ],
        )

    apply(
        i_s,
        j_s,
        u_s,
        v_s,
        ua_s,
        va_s,
        dxc_s,
        dyc_s,
        sin_sg1_s,
        sin_sg2_s,
        sin_sg3_s,
        sin_sg4_s,
        cos_sg1_s,
        cos_sg2_s,
        cos_sg3_s,
        cos_sg4_s,
        rarea_c_s,
        divg_d_s,
        domain=domain(divg_d.shape),
    )

    assert np.allclose(divg_d, divg_d_s)


def horizontal_advection_reference():
    shape = (9, 11, 3)
    rng = np.random.default_rng()
    u = rng.normal(size=shape)
    v = rng.normal(size=shape)
    inp = rng.normal(size=shape)

    inp_xfix = np.copy(inp)
    inp_xfix[:3, :3, :] = np.rot90(inp_xfix[3:6, :3, :], -1)
    inp_xfix[:3, -3:, :] = np.rot90(inp_xfix[3:6, -3:, :], 1)
    inp_xfix[-3:, :3, :] = np.rot90(inp_xfix[-6:-3, :3, :], 1)
    inp_xfix[-3:, -3:, :] = np.rot90(inp_xfix[-6:-3, -3:, :], -1)

    adv_x = u[3:-3, :, :] * (
        -inp_xfix[:-6, :, :]
        + 9 * (inp_xfix[1:-5, :, :] - inp_xfix[5:-1, :, :])
        - 45 * (inp_xfix[2:-4, :, :] - inp_xfix[4:-2, :, :])
        + inp_xfix[6:, :, :]
    ) + np.abs(u[3:-3, :, :]) * (
        -inp_xfix[:-6, :, :]
        + 6 * (inp_xfix[1:-5, :, :] + inp_xfix[5:-1, :, :])
        - 15 * (inp_xfix[2:-4, :, :] + inp_xfix[4:-2, :, :])
        + 20 * inp_xfix[3:-3, :, :]
        - inp_xfix[6:, :, :]
    )

    adv_y = v[3:-3:, 3:-3, :] * (
        -adv_x[:, :-6, :]
        + 9 * (adv_x[:, 1:-5, :] - adv_x[:, 5:-1, :])
        - 45 * (adv_x[:, 2:-4, :] - adv_x[:, 4:-2, :])
        + adv_x[:, 6:, :]
    ) + np.abs(v[3:-3, 3:-3, :]) * (
        -adv_x[:, :-6, :]
        + 6 * (adv_x[:, 1:-5, :] + adv_x[:, 5:-1, :])
        - 15 * (adv_x[:, 2:-4, :] + adv_x[:, 4:-2, :])
        + 20 * adv_x[:, 3:-3, :]
        - adv_x[:, 6:, :]
    )

    return (u, v, inp), adv_y


def advection_impl(i_start, i_end, j_start, j_end):
    def advection(dim):
        @polymorphic_stencil
        def advect(vel, inp):
            return vel[()] * (
                -inp[dim - 3]
                + 9 * (inp[dim - 2] - inp[dim + 2])
                - 45 * (inp[dim - 1] - inp[dim + 1])
                + inp[dim + 3]
            ) + np.abs(vel[()]) * (
                -inp[dim - 3]
                + 6 * (inp[dim - 2] + inp[dim + 2])
                - 15 * (inp[dim - 1] + inp[dim + 1])
                + 20 * inp[()]
                - inp[dim + 3]
            )

        return advect

    @polymorphic_stencil
    def corner_fix_lo_i_lo_j(i, j, inp):
        if j[()] == j_start - 1:
            if i[()] == i_start - 1:
                return inp[I + 1]
            if i[()] == i_start - 2:
                return inp[I + 2, J - 1]
            if i[()] == i_start - 3:
                return inp[I + 3, J - 2]
        if j[()] == j_start - 2:
            if i[()] == i_start - 1:
                return inp[I + 2, J + 1]
            if i[()] == i_start - 2:
                return inp[I + 3]
            if i[()] == i_start - 3:
                return inp[I + 4, J - 1]
        if j[()] == j_start - 3:
            if i[()] == i_start - 1:
                return inp[I + 3, J + 2]
            if i[()] == i_start - 2:
                return inp[I + 4, J + 1]
            if i[()] == i_start - 3:
                return inp[I + 5]
        return inp[()]

    @polymorphic_stencil
    def corner_fix_lo_i_hi_j(i, j, inp):
        if j[()] == j_end:
            if i[()] == i_start - 1:
                return inp[I + 1]
            if i[()] == i_start - 2:
                return inp[I + 2, J + 1]
            if i[()] == i_start - 3:
                return inp[I + 3, J + 2]
        if j[()] == j_end + 1:
            if i[()] == i_start - 1:
                return inp[I + 2, J - 1]
            if i[()] == i_start - 2:
                return inp[I + 3]
            if i[()] == i_start - 3:
                return inp[I + 4, J + 1]
        if j[()] == j_end + 2:
            if i[()] == i_start - 1:
                return inp[I + 3, J - 2]
            if i[()] == i_start - 2:
                return inp[I + 4, J - 1]
            if i[()] == i_start - 3:
                return inp[I + 5]
        return inp[()]

    @polymorphic_stencil
    def corner_fix_hi_i_lo_j(i, j, inp):
        if j[()] == j_start - 1:
            if i[()] == i_end:
                return inp[I - 1]
            if i[()] == i_end + 1:
                return inp[I - 2, J - 1]
            if i[()] == i_end + 2:
                return inp[I - 3, J - 2]
        if j[()] == j_start - 2:
            if i[()] == i_end:
                return inp[I - 2, J + 1]
            if i[()] == i_end + 1:
                return inp[I - 3]
            if i[()] == i_end + 2:
                return inp[I - 4, J - 1]
        if j[()] == j_start - 3:
            if i[()] == i_end:
                return inp[I - 3, J + 2]
            if i[()] == i_end + 1:
                return inp[I - 4, J + 1]
            if i[()] == i_end + 2:
                return inp[I - 5]
        return inp[()]

    @polymorphic_stencil
    def corner_fix_hi_i_hi_j(i, j, inp):
        if j[()] == j_end:
            if i[()] == i_end:
                return inp[I - 1]
            if i[()] == i_end + 1:
                return inp[I - 2, J + 1]
            if i[()] == i_end + 2:
                return inp[I - 3, J + 2]
        if j[()] == j_end + 1:
            if i[()] == i_end:
                return inp[I - 2, J - 1]
            if i[()] == i_end + 1:
                return inp[I - 3]
            if i[()] == i_end + 2:
                return inp[I - 4, J + 1]
        if j[()] == j_end + 2:
            if i[()] == i_end:
                return inp[I - 3, J - 2]
            if i[()] == i_end + 1:
                return inp[I - 4, J - 1]
            if i[()] == i_end + 2:
                return inp[I - 5]
        return inp[()]

    @polymorphic_stencil
    def corner_fix(i, j, inp):
        a = lift(corner_fix_lo_i_lo_j)(i, j, inp)
        b = lift(corner_fix_lo_i_hi_j)(i, j, a)
        c = lift(corner_fix_hi_i_lo_j)(i, j, b)
        return corner_fix_hi_i_hi_j(i, j, c)

    @polymorphic_stencil
    def advection_impl(i, j, u, v, inp):
        xfix = lift(corner_fix)(i, j, inp)
        adv_x = lift(advection(I))(u, xfix)
        return advection(J)(v, adv_x)

    return advection_impl


def test_advection():
    (u, v, inp), out = horizontal_advection_reference()
    u_s = storage(u, origin=(3, 3, 0))
    v_s = storage(v, origin=(3, 3, 0))
    inp_s = storage(inp, origin=(3, 3, 0))
    out_s = storage(np.zeros_like(out))
    i_s = index(inp.shape, "i", origin=(3, 3, 0))
    j_s = index(inp.shape, "j", origin=(3, 3, 0))
    i_start, i_end = 3, inp.shape[0] - 3
    j_start, j_end = 3, inp.shape[1] - 3

    @fencil
    def apply(i, j, u, v, inp, out, domain):
        apply_stencil(
            advection_impl(i_start, i_end, j_start, j_end),
            domain,
            [out],
            [i, j, u, v, inp],
        )

    apply(i_s, j_s, u_s, v_s, inp_s, out_s, domain=domain(out.shape))

    assert np.allclose(out, out_s)
