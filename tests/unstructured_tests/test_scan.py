from unstructured.concepts import (
    backward_scan,
    element_access_to_field,
    generic_scan,
    scan_pass,
)
from unstructured.utils import (
    axis,
    split_indices,
)
import numpy as np
from unstructured.helpers import array_as_field


@axis(length=5)
class K:
    ...


@axis(length=2)
class I:
    ...


def k_sum_explicit(inp):
    @element_access_to_field(
        axises=inp.axises, element_type=inp.element_type, tuple_size=None
    )
    def elem_acc(indices):
        state = 0
        k_index, rest = split_indices(indices, (K,))
        assert len(k_index) == 1
        for k in map(lambda i: K(i), range(k_index[0])):
            state = state + inp[k]
        return state[rest]

    return elem_acc


def test_explicit_scan():
    field = array_as_field(I, K)(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))

    out = k_sum_explicit(field)

    assert out[K(1), I(0)] == 1
    assert out[K(4), I(0)] == 10
    assert out[K(1), I(1)] == 6


test_explicit_scan()


def k_sum(state, inp):  # both state and inp are k slices
    if state is None:
        res = inp
    else:
        res = state + inp
    return res


def k_sum_scanner(inp):
    @element_access_to_field(
        axises=inp.axises, element_type=inp.element_type, tuple_size=None
    )
    def elem_acc(indices):
        k_index, rest = split_indices(indices, (K,))
        assert len(k_index) == 1

        state = None
        for k in map(lambda i: K(i), range(k_index[0])):
            state = k_sum(state, inp[k])
        return state[rest]

    return elem_acc


def test_scan():
    field = array_as_field(I, K)(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))

    out = k_sum_scanner(field)

    assert out[K(1), I(0)] == 1
    assert out[K(4), I(0)] == 10
    assert out[K(1), I(1)] == 6


test_scan()


def test_generic_scan():
    field = array_as_field(I, K)(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))

    out = generic_scan(K, k_sum, backward=False)(field)

    assert out[K(1), I(0)] == 3
    assert out[K(4), I(0)] == 15
    assert out[K(1), I(1)] == 13


test_generic_scan()


def k_sum_2inp(state, inp1, inp2):
    if state is None:
        res = inp1 + inp2
    else:
        res = state + inp1 + inp2
    return res


def test_2inp():
    field = array_as_field(I, K)(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))

    out = generic_scan(K, k_sum_2inp, backward=False)(field, field)

    assert out[K(1), I(0)] == 6
    assert out[K(4), I(0)] == 30
    assert out[K(1), I(1)] == 26


test_2inp()


def k_sum_2inp_2out(state, inp1, inp2):
    if state is None:
        res1 = inp1
        res2 = inp2
    else:
        s1, s2 = state
        res1 = s1 + inp1
        res2 = s2 + inp2
    return res1, res2


def test_2out():
    field1 = array_as_field(I, K)(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))
    field2 = array_as_field(I, K)(
        np.array([[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]])
    )

    out1, out2 = generic_scan(K, k_sum_2inp_2out, backward=False)(field1, field2)

    assert out1[K(1), I(0)] == 3
    assert out1[K(4), I(0)] == 15
    assert out1[K(1), I(1)] == 13

    assert out2[K(1), I(0)] == 23
    assert out2[K(2), I(0)] == 36
    assert out2[K(2), I(1)] == 51


test_2out()


@scan_pass(K)
def decorated_k_sum(state, inp):  # both state and inp are fields without k dimension
    if state is None:
        res = inp
    else:
        res = state + inp
    return res


def test_decorated():
    field = array_as_field(I, K)(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))

    out = decorated_k_sum(field)

    assert out[K(1), I(0)] == 3
    assert out[K(4), I(0)] == 15
    assert out[K(1), I(1)] == 13


test_decorated()


@backward_scan(K)
def backward_k_sum(state, inp):
    if state is None:
        res = inp
    else:
        res = state + inp
    return res


def test_backward():
    field = array_as_field(I, K)(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))

    out = backward_k_sum(field)

    assert out[K(1), I(0)] == 14
    assert out[K(4), I(0)] == 5
    assert out[K(1), I(1)] == 34


test_backward()


def test_forward_backward():
    field = array_as_field(I, K)(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))
    # forward result: [1,3,6,10,15], [6,13,21,30,40]

    out = backward_k_sum(decorated_k_sum(field))

    assert out[K(1), I(0)] == 34
    assert out[K(4), I(0)] == 15
    assert out[K(1), I(1)] == 104
