from unstructured.concepts import element_access_to_field
from unstructured.utils import axis, get_index_of_type, split_indices
import numpy as np
from unstructured.helpers import array_as_field


@axis(length=5)
class K:
    ...


@axis(length=2)
class I:
    ...


def k_sum_explicit(inp):
    @element_access_to_field(axises=inp.axises, element_type=inp.element_type)
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
    @element_access_to_field(axises=inp.axises, element_type=inp.element_type)
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


def generic_scan(axis, fun):
    def scanner(*inps):
        # TODO assert all inp have the same axises
        @element_access_to_field(
            axises=inps[0].axises, element_type=inps[0].element_type
        )
        def elem_acc(indices):
            scan_index, rest = split_indices(indices, (axis,))
            assert len(scan_index) == 1

            state = None
            for ind in map(lambda i: axis(i), range(scan_index[0])):
                state = fun(state, *tuple(inp[ind] for inp in inps))
            return state[rest]

        return elem_acc

    return scanner


def test_generic_scan():
    field = array_as_field(I, K)(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))

    out = generic_scan(K, k_sum)(field)

    assert out[K(1), I(0)] == 1
    assert out[K(4), I(0)] == 10
    assert out[K(1), I(1)] == 6


test_generic_scan()


def k_sum_2inp(state, inp1, inp2):
    if state is None:
        res = inp1 + inp2
    else:
        res = state + inp1 + inp2
    return res


def test_2inp():
    field = array_as_field(I, K)(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))

    out = generic_scan(K, k_sum_2inp)(field, field)

    assert out[K(1), I(0)] == 2
    assert out[K(4), I(0)] == 20
    assert out[K(1), I(1)] == 12


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

    out1, out2 = generic_scan(K, k_sum_2inp_2out)(field1, field2)

    assert out1[K(1), I(0)] == 1
    assert out1[K(4), I(0)] == 10
    assert out1[K(1), I(1)] == 6

    assert out2[K(1), I(0)] == 11
    assert out2[K(2), I(0)] == 23
    assert out2[K(2), I(1)] == 33


test_2inp()
