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
        (0.0, 0.0, qg + qsum) if qsum <= 0.0 else
        (0.0, qsum, qg) if qi < 0.0 else
        (qsum, 0.0, qg) if qs < 0.0 else
        (qi, qs, qg)
    )


def ifs1_fun2(qx, qg):
    dq = qx if qx < -qg else -qg
    return qx - dq, qg + dq


def ifs1_fun3(qi, qs, qg):
    qs1, qg1 = ifs1_fun2(qs, qg)
    qi1, qg2 = ifs1_fun2(qi, qg1)
    return (
        (qi, qs, qg) if qg >= 0.0 else
        (qi, qs1, qg1) if qg1 >= 0.0 else
        (qi1, qs1, qg2)
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
