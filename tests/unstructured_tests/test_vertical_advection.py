from unstructured.concepts import apply_stencil, backward_scan, forward_scan
from unstructured.utils import axis, print_axises
from unstructured.helpers import array_as_field
import numpy as np
import pytest


@axis(length=5)
class KDim:
    ...


@axis()
class IDim:
    ...


@axis()
class JDim:
    ...


@forward_scan(KDim)
def tridiag_forward(state, a, b, c, d):
    if state is None:
        cp_k = c / b
        dp_k = d / b
    else:
        cp_km1, dp_km1 = state
        cp_k = c / (b - a * cp_km1)
        dp_k = (d - a * dp_km1) / (b - a * cp_km1)
    return cp_k, dp_k


@backward_scan(KDim)
def tridiag_backward(x_kp1, cp, dp):
    if x_kp1 is None:
        x_k = dp
    else:
        x_k = dp - cp * x_kp1
    return x_k


def solve_tridiag(a, b, c, d):
    cp, dp = tridiag_forward(a, b, c, d)
    return tridiag_backward(cp, dp)


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


def test_tridiag(tridiag_reference):
    a, b, c, d, x = tridiag_reference
    shape = a.shape
    assert shape[2] == len(KDim(0))
    as_3d_field = array_as_field(IDim, JDim, KDim)
    a_s = as_3d_field(a)
    b_s = as_3d_field(b)
    c_s = as_3d_field(c)
    d_s = as_3d_field(d)
    x_s = np.zeros_like(x)

    apply_stencil(
        solve_tridiag,
        [(range(shape[0]), IDim), (range(shape[1]), JDim), (range(shape[2]), KDim)],
        [a_s, b_s, c_s, d_s],
        [x_s],
    )

    assert np.allclose(x, np.asarray(x_s))
