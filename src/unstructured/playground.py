# MDIterator
#  - has `deref()`
#  - has `position`, dict of key (`axis`) to index

# shift(Iterator, GeneralizedOffset) -> Iterator

# GeneralizedOffset is a callable that takes pos and returns new pos


from typing import Callable, Dict
import numpy as np

# concepts


class MDIterator:
    def __init__(self, derefer, pos) -> None:
        self.derefer = derefer
        self.pos = pos

    def deref(self):
        return self.derefer(*self.pos.values())


def shift(iter, offset):
    return MDIterator(iter.derefer, offset(iter.pos))


def deref(iter):
    return iter.deref()


# helpers


def np_as_iterator(*axises):
    def _maker(a):
        return MDIterator(lambda *indices: a[indices], {axis: 0 for axis in axises})

    return _maker


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


def test_iterator():
    inp = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    inpit = np_as_iterator(I, J)(inp)
    print(inpit.deref())
    shifted = shift(inpit, I(1))
    print(shifted.deref())
    shifted = shift(shifted, J(2))
    print(shifted.deref())


test_iterator()


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


# class IStag:
#     def __init__(self, offset) -> None:
#         self.offset = offset

#     def __call__(self, pos: Dict) -> Dict:
#         if I in pos.keys():
#             new_pos = pos.copy()
#             ipos = int(new_pos[I] + self.offset + 0.5)
#             del new_pos[I]
#             new_pos[IStag] = ipos
#             return new_pos
#         return pos


def stag_sum(inp_stag: IStag, inp: I) -> IStag:
    return (
        deref(inp_stag)
        + deref(shift(inp, IStag2I(-0.5)))
        + deref(shift(inp, IStag2I(0.5)))
    )


def test_stag():
    inp = np.asarray([1, 2, 3, 4, 5])
    inp_stag = np.asarray([10, 20, 30, 40])
    res = np.asarray([13, 25, 37, 49])

    inp_it = np_as_iterator(I)(inp)
    inp_stag_it = np_as_iterator(IStag)(inp_stag)

    # move the iterator in the non staggered field to the staggered origin: (i.e. move the iterator of all fields to the same physical location)
    # TODO seems to work for staggering (because its just a single offset), but not for unstructured, i.e. vertices and edges
    inp_it = shift(inp_it, I2IStag(0.5))

    # this is the apply stencil explicit
    print(stag_sum(inp_stag_it, inp_it))

    inp_it = shift(inp_it, IStag(1))
    inp_stag_it = shift(inp_stag_it, IStag(1))
    print(stag_sum(inp_stag_it, inp_it))

    inp_it = shift(inp_it, IStag(1))
    inp_stag_it = shift(inp_stag_it, IStag(1))
    print(stag_sum(inp_stag_it, inp_it))

    inp_it = shift(inp_it, IStag(1))
    inp_stag_it = shift(inp_stag_it, IStag(1))
    print(stag_sum(inp_stag_it, inp_it))


test_stag()
