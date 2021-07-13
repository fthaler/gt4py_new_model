import itertools
from typing_extensions import runtime
import unstructured
from unstructured.builtins import builtin_dispatch, lift, shift, deref, cartesian, if_
from unstructured.runtime import closure, offset
from unstructured.utils import tupelize
import numpy as np

EMBEDDED = "embedded"

# def _shift_impl(pos, offset):
#     if isinstance(offset, St)
class CartesianAxis:
    ...


class I_loc(
    CartesianAxis
):  # make user definable, requires domain builtin to take these axis keys
    ...


class J_loc(CartesianAxis):
    ...


@shift.register(EMBEDDED)
def shift(*offsets):
    def impl(iter):
        return iter.shift(
            *offsets
        )  # could be removed as only during execution we know what shift does

    return impl
    # raise RuntimeError("shift is not configured")


@deref.register(EMBEDDED)
def deref(iter):
    return iter.deref()


@if_.register(EMBEDDED)
def if_(cond, t, f):
    return t if cond else f


@lift.register(EMBEDDED)
def lift(stencil):
    def impl(*args):
        class wrap_iterator:
            def __init__(self, *, offsets=[]) -> None:
                self.offsets = offsets

            def shift(self, offset):
                return wrap_iterator(offsets=[*self.offsets, offset])

            # def get_max_number_of_neighbors(self):
            #     for o in reversed(self.offsets):
            #         if isinstance(o, OffsetGroup):
            #             return len(o.offsets)
            #     raise RuntimeError("Error")

            def deref(self):
                shifted_args = args
                for offset in self.offsets:
                    shifted_args = tuple(
                        map(lambda arg: arg.shift(offset), shifted_args)
                    )

                if any(shifted_arg.is_none() for shifted_arg in shifted_args):
                    return None
                return stencil(*shifted_args)

        return wrap_iterator()

    return impl


@cartesian.register(EMBEDDED)
def cartesian(is_, ie, js, je):
    return {I_loc: range(is_, ie), J_loc: range(js, je)}


def named_range(axis, range):
    return ((axis, i) for i in range)


def domain_iterator(domain):
    return (
        dict(elem)
        for elem in itertools.product(
            *map(lambda tup: named_range(tup[0], tup[1]), domain.items())
        )
    )


def execute_shift(pos, tag, index):
    if issubclass(tag, CartesianAxis):
        assert tag in pos
        new_pos = pos.copy()
        new_pos[tag] += index
        return new_pos
    assert False


# The following holds for shifts:
# shift(tag, index)(inp) -> full shift
# shift(tag)(inp) -> incomplete shift
# shift(index)(shift(tag)(inp)) -> full shift
# Therefore the following transformation holds
# shift(e2c,0)(shift(v2c,2)(cell_field))
# = shift(0)(shift(e2c)(shift(2)(shift(v2c)(cell_field))))
# = shift(v2c, 2, e2c, 0)(cell_field)
# = shift(v2c,e2c,2,0)(cell_field) <-- v2c,e2c twice incomplete shift
# = shift(2,0)(shift(v2c,e2c)(cell_field))
# for implementations it means everytime we have an index, we can "execute" a concrete shift
def shift_position(pos, *offsets):
    new_pos = pos
    tag_stack = []
    for offset in offsets:
        if isinstance(offset, int):
            assert tag_stack
            tag = tag_stack.pop(0)  # first tag corresponds to current index
            new_pos = execute_shift(new_pos, tag, offset)
            if new_pos is None:
                return None
        else:
            tag_stack.append(offset)
    assert not tag_stack
    return new_pos


class MDIterator:
    def __init__(self, field, pos, *, offsets=[]) -> None:
        self.field = field
        self.pos = pos
        self.offsets = offsets

    def shift(self, *offsets):
        return MDIterator(self.field, self.pos, offsets=[*self.offsets, *offsets])

    # def get_max_number_of_neighbors(self):
    #     return len(self.get_shifted_pos()[NeighborAxis][-1].offsets)

    def get_shifted_pos(self):
        return shift_position(self.pos, *self.offsets)
        # shifted_pos = self.pos
        # if shifted_pos is None:
        #     return None
        # for offset in self.offsets:
        #     shifted_pos = _shift_impl(shifted_pos, offset)
        #     if shifted_pos is None:
        #         return None
        # return shifted_pos

    def is_none(self):
        return self.get_shifted_pos() is None

    def deref(self):
        shifted_pos = self.get_shifted_pos()

        if not all(
            axis in [*shifted_pos.keys()]
            for axis in self.field.axises
            # axis in [*shifted_pos.keys(), ExtraDim] for axis in self.field.axises
        ):
            raise IndexError(
                "Iterator position doesn't point to valid location for its field."
            )
        ordered_indices = get_ordered_indices(self.field.axises, shifted_pos)
        return self.field[ordered_indices]


def make_in_iterator(inp, pos):
    return MDIterator(inp, pos)
    # neighbor_axis = _get_neighbor_axis(inp.axises)
    # if neighbor_axis is None:
    #     return MDIterator(inp, pos)
    # else:
    #     new_pos = pos.copy()
    #     new_pos[neighbor_axis] = 0
    #     return MDIterator(inp, new_pos)


@closure.register(EMBEDDED)
def closure(domain, sten, outs, ins):  # domain is Dict[axis, range]
    for pos in domain_iterator(domain):
        # ins_iters = list(MDIterator(inp, pos) for inp in ins)
        ins_iters = list(make_in_iterator(inp, pos) for inp in ins)
        res = sten(*ins_iters)
        if not isinstance(res, tuple):
            res = (res,)
        if not len(res) == len(outs):
            IndexError("Number of return values doesn't match number of output fields.")

        for r, out in zip(res, outs):
            ordered_indices = tuple(get_ordered_indices(out.axises, pos))
            out[ordered_indices] = r


builtin_dispatch.push_key(EMBEDDED)  # makes embedded the default


class LocatedField:
    """A Field with named dimensions/axises.

    Axis keys can be any objects that are hashable.
    """

    def __init__(self, getter, axises, *, setter=None, array=None):
        self.getter = getter
        self.axises = axises
        self.setter = setter
        self.array = array

    def __getitem__(self, indices):
        indices = tupelize(indices)
        return self.getter(indices)

    def __setitem__(self, indices, value):
        if self.setter is None:
            raise TypeError("__setitem__ not supported for this field")
        self.setter(indices, value)

    def __array__(self):
        if self.array is None:
            raise TypeError("__array__ not supported for this field")
        return self.array()


def get_ordered_indices(axises, pos):
    """pos is a dictionary from axis to offset"""
    assert all(axis in pos for axis in axises)
    # return tuple(slice(None) if axis is ExtraDim else pos[axis] for axis in axises)
    return tuple(pos[axis] for axis in axises)


def _tupsum(a, b):
    return tuple(sum(i) for i in zip(a, b))
    # def sum_if_not_slice(ab_elems):
    #     return (
    #         slice(None)
    #         if (isinstance(ab_elems[0], slice) or isinstance(ab_elems[1], slice))
    #         else sum(ab_elems)
    #     )

    # return tuple(sum_if_not_slice(i) for i in zip(a, b))


def np_as_located_field(*axises, origin=None):
    def _maker(a: np.ndarray):
        if a.ndim != len(axises):
            raise TypeError("ndarray.ndim incompatible with number of given axises")

        if origin is not None:
            offsets = get_ordered_indices(axises, origin)
        else:
            offsets = tuple(0 for _ in axises)

        def setter(indices, value):
            a[_tupsum(indices, offsets)] = value

        def getter(indices):
            return a[_tupsum(indices, offsets)]

        return LocatedField(getter, axises, setter=setter, array=a.__array__)

    return _maker


from unstructured.builtins import *
from unstructured.runtime import *


def fendef_embedded(fun, *args, **kwargs):
    assert "offset_provider" in kwargs

    @unstructured.builtins.shift.register(EMBEDDED)
    def shift(*offsets):
        def impl(iter):
            return iter.shift(
                *[
                    offset
                    if isinstance(offset, int)
                    else kwargs["offset_provider"][offset.value]
                    for offset in offsets
                ]
            )

        return impl

    fun(*args)


unstructured.runtime.fundef_registry[None] = fendef_embedded

# @unstructured.runtime.fendef.register(EMBEDDED)
# def fendef(offset_provider):
#     def impl(fun):
#         def impl2(*args):
#             @unstructured.builtins.shift.register(EMBEDDED)
#             def shift(*offsets):
#                 def impl(iter):
#                     return iter.shift(
#                         *[
#                             offset
#                             if isinstance(offset, int)
#                             else offset_provider[offset.value]
#                             for offset in offsets
#                         ]
#                     )

#                 return impl

#             fun(*args)

#         return impl2

#     return impl


# @unstructured.runtime.fundef.register(EMBEDDED)
# def fundef(fun):
#     return fun


# unstructured.runtime.fun_fen_def_dispatch.push_key(EMBEDDED)
