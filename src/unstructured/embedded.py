import itertools

import unstructured
from unstructured.builtins import (
    builtin_dispatch,
    lift,
    reduce,
    shift,
    deref,
    domain,
    named_range,
    if_,
    minus,
    plus,
    mul,
    div,
    greater,
)
from unstructured.runtime import CartesianAxis, Offset
from unstructured.utils import tupelize
import numpy as np

EMBEDDED = "embedded"


class NeighborTableOffsetProvider:
    def __init__(self, tbl, origin_axis, neighbor_axis, max_neighbors) -> None:
        self.tbl = tbl
        self.origin_axis = origin_axis
        self.neighbor_axis = neighbor_axis
        self.max_neighbors = max_neighbors


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

            def shift(self, *offsets):
                return wrap_iterator(offsets=[*offsets, *self.offsets])

            def max_neighbors(self):
                # TODO cleanup, test edge cases
                open_offsets = get_open_offsets(*self.offsets)
                assert open_offsets
                assert isinstance(
                    args[0].offset_provider[open_offsets[0].value],
                    NeighborTableOffsetProvider,
                )
                return args[0].offset_provider[open_offsets[0].value].max_neighbors

            def deref(self):
                class DelayedIterator:
                    def __init__(
                        self, wrapped_iterator, lifted_offsets, *, offsets=[]
                    ) -> None:
                        self.wrapped_iterator = wrapped_iterator
                        self.lifted_offsets = lifted_offsets
                        self.offsets = offsets

                    def is_none(self):
                        shifted = self.wrapped_iterator.shift(
                            *self.lifted_offsets, *self.offsets
                        )
                        return shifted.is_none()

                    def shift(self, *offsets):
                        return DelayedIterator(
                            self.wrapped_iterator,
                            self.lifted_offsets,
                            offsets=[*offsets, *self.offsets],
                        )

                    def deref(self):
                        shifted = self.wrapped_iterator.shift(
                            *self.lifted_offsets, *self.offsets
                        )
                        return shifted.deref()

                shifted_args = tuple(
                    map(lambda arg: DelayedIterator(arg, self.offsets), args)
                )

                if any(shifted_arg.is_none() for shifted_arg in shifted_args):
                    return None
                return stencil(*shifted_args)

        return wrap_iterator()

    return impl


@reduce.register(EMBEDDED)
def reduce(fun, init):
    def sten(*iters):
        # assert check_that_all_iterators_are_compatible(*iters)
        first_it = iters[0]
        n = first_it.max_neighbors()
        res = init
        for i in range(n):
            # we can check a single argument
            # because all arguments share the same pattern
            if (
                unstructured.builtins.deref(unstructured.builtins.shift(i)(first_it))
                is None
            ):
                break
            res = fun(
                res,
                *(
                    unstructured.builtins.deref(unstructured.builtins.shift(i)(it))
                    for it in iters
                )
            )
        return res

    return sten


@domain.register(EMBEDDED)
def domain(*args):
    domain = {}
    for arg in args:
        domain.update(arg)
    return domain


@named_range.register(EMBEDDED)
def named_range(tag, start, end):
    return {tag: range(start, end)}


@minus.register(EMBEDDED)
def minus(first, second):
    return first - second


@plus.register(EMBEDDED)
def plus(first, second):
    return first + second


@mul.register(EMBEDDED)
def mul(first, second):
    return first * second


@div.register(EMBEDDED)
def div(first, second):
    return first / second


@greater.register(EMBEDDED)
def greater(first, second):
    return first > second


def named_range(axis, range):
    return ((axis, i) for i in range)


def domain_iterator(domain):
    return (
        dict(elem)
        for elem in itertools.product(
            *map(lambda tup: named_range(tup[0], tup[1]), domain.items())
        )
    )


def execute_shift(pos, tag, index, *, offset_provider):
    if (
        tag in pos and pos[tag] is None
    ):  # sparse field with offset as neighbor dimension
        new_pos = pos.copy()
        new_pos[tag] = index
        return new_pos
    assert tag.value in offset_provider
    offset_implementation = offset_provider[tag.value]
    if isinstance(offset_implementation, CartesianAxis):
        assert offset_implementation in pos
        new_pos = pos.copy()
        new_pos[offset_implementation] += index
        return new_pos
    elif isinstance(offset_implementation, NeighborTableOffsetProvider):
        assert offset_implementation.origin_axis in pos
        new_pos = pos.copy()
        del new_pos[offset_implementation.origin_axis]
        if (
            offset_implementation.tbl[pos[offset_implementation.origin_axis], index]
            is None
        ):
            return None
        else:
            new_pos[offset_implementation.neighbor_axis] = offset_implementation.tbl[
                pos[offset_implementation.origin_axis], index
            ]
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
def group_offsets(*offsets):
    tag_stack = []
    index_stack = []
    complete_offsets = []
    for offset in offsets:
        if not isinstance(offset, int):
            if index_stack:
                index = index_stack.pop(0)
                complete_offsets.append((offset, index))
            else:
                tag_stack.append(offset)
        else:
            assert not tag_stack
            index_stack.append(offset)
    return complete_offsets, tag_stack


def shift_position(pos, *offsets, offset_provider):
    complete_offsets, open_offsets = group_offsets(*offsets)
    assert not open_offsets

    new_pos = pos
    for tag, index in complete_offsets:
        new_pos = execute_shift(new_pos, tag, index, offset_provider=offset_provider)
        if new_pos is None:
            return None
    return new_pos


def get_open_offsets(*offsets):
    return group_offsets(*offsets)[1]


class MDIterator:
    def __init__(self, field, pos, *, offsets=[], offset_provider) -> None:
        self.field = field
        self.pos = pos
        self.offsets = offsets
        self.offset_provider = offset_provider

    def shift(self, *offsets):
        return MDIterator(
            self.field,
            self.pos,
            offsets=[*offsets, *self.offsets],
            offset_provider=self.offset_provider,
        )

    def max_neighbors(self):
        open_offsets = get_open_offsets(*self.offsets)
        assert open_offsets
        assert isinstance(
            self.offset_provider[open_offsets[0].value], NeighborTableOffsetProvider
        )
        return self.offset_provider[open_offsets[0].value].max_neighbors

    def is_none(self):
        return (
            shift_position(
                self.pos, *self.offsets, offset_provider=self.offset_provider
            )
            is None
        )

    def deref(self):
        shifted_pos = shift_position(
            self.pos, *self.offsets, offset_provider=self.offset_provider
        )

        if not all(axis in [*shifted_pos.keys()] for axis in self.field.axises):
            raise IndexError(
                "Iterator position doesn't point to valid location for its field."
            )
        ordered_indices = get_ordered_indices(self.field.axises, shifted_pos)
        return self.field[ordered_indices]


def make_in_iterator(inp, pos, offset_provider):
    sparse_dimensions = [axis for axis in inp.axises if isinstance(axis, Offset)]
    assert len(sparse_dimensions) <= 1  # TODO multiple is not a current use case
    new_pos = pos.copy()
    for axis in sparse_dimensions:
        new_pos[axis] = None
    return MDIterator(
        inp, new_pos, offsets=[*sparse_dimensions], offset_provider=offset_provider
    )


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


def index_field(axis):
    return LocatedField(lambda index: index[0], (axis,))


@unstructured.builtins.shift.register(EMBEDDED)
def shift(*offsets):
    def impl(iter):
        return iter.shift(*reversed(offsets))

    return impl


def fendef_embedded(fun, *args, **kwargs):
    assert "offset_provider" in kwargs

    @unstructured.runtime.closure.register(EMBEDDED)
    def closure(domain, sten, outs, ins):  # domain is Dict[axis, range]
        for pos in domain_iterator(domain):
            ins_iters = list(
                make_in_iterator(inp, pos, kwargs["offset_provider"]) for inp in ins
            )
            res = sten(*ins_iters)
            if not isinstance(res, tuple):
                res = (res,)
            if not len(res) == len(outs):
                IndexError(
                    "Number of return values doesn't match number of output fields."
                )

            for r, out in zip(res, outs):
                ordered_indices = tuple(get_ordered_indices(out.axises, pos))
                out[ordered_indices] = r

    fun(*args)


unstructured.runtime.fendef_registry[None] = fendef_embedded
