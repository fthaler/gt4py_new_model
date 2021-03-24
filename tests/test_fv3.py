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
