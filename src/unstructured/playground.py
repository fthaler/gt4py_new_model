# MDIterator
#  - has `deref()`
#  - has `position`, dict of key (`axis`) to index

# shift(Iterator, GeneralizedOffset) -> Iterator

# GeneralizedOffset is a callable that takes pos and returns new pos


from typing import Callable, Dict
import numpy as np
import pytest
import itertools

from unstructured.utils import tupelize

# concepts


def get_order_indices(axises, pos):
    return tuple(pos[axis] for axis in axises)


class MDIterator:
    def __init__(self, field, pos) -> None:
        self.field = field
        self.pos = pos

    def deref(self):
        if not all(axis in self.pos.keys() for axis in self.field.axises):
            raise IndexError(
                "Iterator position doesn't point to valid location for its field."
            )
        ordered_indices = get_order_indices(self.field.axises, self.pos)
        return self.field[ordered_indices]


class LocatedField:
    def __init__(self, getter, axises, *, setter=None):
        self.getter = getter
        self.axises = axises
        self.setter = setter

    def __getitem__(self, indices):
        indices = tupelize(indices)
        return self.getter(indices)

    def __setitem__(self, indices, value):
        if self.setter is None:
            raise TypeError("__setitem__ not supported for this field")
        self.setter(indices, value)


def shift(iter, offset):
    return MDIterator(iter.field, offset(iter.pos))


def deref(iter):
    return iter.deref()


def named_range(axis, range):
    return ((axis, i) for i in range)


def domain_iterator(domain):
    return (
        dict(elem)
        for elem in itertools.product(
            *map(lambda tup: named_range(tup[0], tup[1]), domain.items())
        )
    )


def apply_stencil(sten, domain, ins, outs):  # domain is Dict[axis, range]
    for pos in domain_iterator(domain):
        ins_iters = list(MDIterator(inp, pos) for inp in ins)
        res = sten(*ins_iters)
        if not isinstance(res, tuple):
            res = (res,)
        if not len(res) == len(outs):
            IndexError("Number of return values doesn't match number of output fields.")

        for r, out in zip(res, outs):
            ordered_indices = tuple(get_order_indices(out.axises, pos))
            out[ordered_indices] = r


# helpers


def _tupsum(a, b):
    return tuple(sum(i) for i in zip(a, b))


def np_as_located_field(*axises, origin=None):
    def _maker(a: np.ndarray):
        if a.ndim != len(axises):
            raise TypeError("ndarray.ndim incompatible with number of given axises")

        if origin is not None:
            offsets = get_order_indices(axises, origin)
        else:
            offsets = tuple(0 for _ in axises)

        def setter(indices, value):
            a[_tupsum(indices, offsets)] = value

        def getter(indices):
            return a[_tupsum(indices, offsets)]

        return LocatedField(getter, axises, setter=setter)

    return _maker


# tests
def test_domain():
    class I:
        ...

    class J:
        ...

    domain = {I: range(2), J: range(3)}
    for pos in domain_iterator(domain):
        print(pos)


test_domain()

# user code


class CartesianAxis:
    def __init__(self, offset) -> None:
        self.offset = offset

    def __call__(self, pos: Dict) -> Dict:
        axis = type(self)
        if axis in pos.keys():
            new_pos = pos.copy()
            new_pos[axis] += self.offset
            return new_pos
        return pos


class I(CartesianAxis):
    ...


class J(CartesianAxis):
    ...


class K(CartesianAxis):
    ...


def test_iterator():
    inp = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    inp_located = np_as_located_field(I, J)(inp)

    inp_located[2, 3] = 20
    assert inp_located[2, 3] == 20

    inp_it = MDIterator(inp_located, {I: 2, J: 0})
    print(deref(inp_it))
    assert deref(MDIterator(inp_located, {I: 2, J: 0})) == deref(
        MDIterator(inp_located, {J: 0, I: 2})
    )

    with pytest.raises(IndexError):
        deref(MDIterator(inp_located, {I: 0, K: 0}))


test_iterator()


def test_shift():
    inp = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    inp_located = np_as_located_field(I, J)(inp)
    inp_it = MDIterator(inp_located, {I: 1, J: 2})

    res = shift(shift(inp_it, I(1)), J(-1))
    assert deref(res) == 10


test_shift()


class IStag(CartesianAxis):
    ...


class I2IStag:
    def __init__(self, offset) -> None:
        self.offset = offset

    def __call__(self, pos: Dict) -> Dict:
        if I in pos.keys():
            new_pos = pos.copy()
            ipos = int(new_pos[I] + self.offset - 0.5)
            del new_pos[I]
            new_pos[IStag] = ipos
            return new_pos
        return pos


class IStag2I:
    def __init__(self, offset) -> None:
        self.offset = offset

    def __call__(self, pos: Dict) -> Dict:
        if IStag in pos.keys():
            new_pos = pos.copy()
            ipos = int(new_pos[IStag] + self.offset + 0.5)
            del new_pos[IStag]
            new_pos[I] = ipos
            return new_pos
        return pos


def stag_sum(inp_stag, inp):
    return (
        deref(inp_stag)
        + deref(shift(inp, IStag2I(-0.5)))
        + deref(shift(inp, IStag2I(0.5)))
    )


def broken_stag(inp):
    return deref(inp)


def test_stag():
    inp = np_as_located_field(I)(np.asarray([1, 2, 3, 4, 5]))
    inp_stag = np_as_located_field(IStag)(np.asarray([10, 20, 30, 40]))
    res = np.asarray([13, 25, 37, 49])

    out = np_as_located_field(IStag)(np.zeros([4]))

    # this is the apply stencil explicit
    inp_it = MDIterator(inp, {IStag: 0})
    inp_stag_it = MDIterator(inp_stag, {IStag: 0})
    print(stag_sum(inp_stag_it, inp_it))

    apply_stencil(stag_sum, {IStag: range(4)}, [inp_stag, inp], [out])
    print(out[0])
    print(out[1])
    print(out[2])
    print(out[3])

    with pytest.raises(IndexError):
        apply_stencil(broken_stag, {IStag: range(4)}, [inp], [out])


test_stag()


def lr_sum(inp):
    return deref(shift(inp, I(-1))) + deref(shift(inp, I(1)))


def test_non_zero_origin():
    inp = np_as_located_field(I, origin={I: 1})(np.asarray([1, 2, 3, 4, 5]))
    out = np_as_located_field(I)(np.zeros([3]))
    apply_stencil(lr_sum, {I: range(3)}, [inp], [out])
    assert out[0] == 4
    assert out[1] == 6
    assert out[2] == 8


test_non_zero_origin()


def explicit_nested_sum(inp):
    class shiftable_sum:
        shift = None

        def deref(self):
            return deref(shift(inp, I(-1))) + deref(shift(inp, I(1)))

    return deref(shiftable_sum())


def test_explicit_nested_sum():
    inp = np_as_located_field(I, origin={I: 1})(np.asarray([1, 2, 3, 4, 5]))
    out = np_as_located_field(I)(np.zeros([3]))
    apply_stencil(explicit_nested_sum, {I: range(3)}, [inp], [out])
    print(out[0])


test_explicit_nested_sum()

# def lift(stencil):
#     def impl(arg):  # TODO multiple args
#         class _field:

#         wrap = MDIterator(arg.field, arg.pos)
#         return wrap

#     return impl


# def nested_sum(inp):
#     sum = lift(lr_sum)(inp)
#     return lr_sum(sum)


# def test_lift():
#     inp = np_as_located_field(I, origin={I: 2})(np.asarray([1, 2, 3, 4, 5]))
#     out = np_as_located_field(I)(np.zeros([1]))
#     apply_stencil(nested_sum, {I: range(1)}, [inp], [out])
#     print(out[0])
#     assert out[0] == 12


# test_lift()
