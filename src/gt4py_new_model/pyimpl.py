from collections.abc import Callable, Mapping, Set
from dataclasses import dataclass
import operator
from typing import Any, Generic, Union, TypeVar

import numpy as np


@dataclass(frozen=True)
class AxisOffset:
    axis: int
    offset: int

    def __add__(self, other):
        if isinstance(other, AxisOffset):
            assert other.axis == self.axis
            return AxisOffset(axis=self.axis, offset=self.offset + other.offset)
        return AxisOffset(axis=self.axis, offset=self.offset + other)

    def __sub__(self, other):
        if isinstance(other, AxisOffset):
            assert other.axis == self.axis
            return AxisOffset(axis=self.axis, offset=self.offset - other.offset)
        return AxisOffset(axis=self.axis, offset=self.offset - other)

    def collect(*indices):
        res = dict()
        for i in indices:
            res[i.axis] = i + res.get(i.axis, AxisOffset(axis=i.axis, offset=0))
        return tuple(res.values())


I = AxisOffset(axis=0, offset=0)
J = AxisOffset(axis=1, offset=0)
K = AxisOffset(axis=2, offset=0)

Accessor = Mapping[Union[AxisOffset, tuple[AxisOffset, ...]], Any]


@dataclass(frozen=True)
class LiftedAccessor:
    accessor: Accessor
    offsets: tuple[AxisOffset, ...]

    def __getitem__(self, offsets):
        if isinstance(offsets, AxisOffset):
            offsets = (offsets,)
        return LiftedAccessor(
            accessor=self.accessor, offsets=AxisOffset.collect(*self.offsets, *offsets)
        )

    def _evaluate(self):
        return self.accessor[self.offsets]

    def _op(self, other, op):
        class Accessor:
            def __getitem__(_, offsets):
                if isinstance(other, LiftedAccessor):
                    other_offsets = AxisOffset.collect(*other.offsets, *offsets)
                    other_value = other.accessor[other_offsets]
                else:
                    other_value = other
                self_offsets = AxisOffset.collect(*self.offsets, *offsets)
                return op(self.accessor[self_offsets], other_value)

        return LiftedAccessor(accessor=Accessor(), offsets=())

    def _rop(self, other, op):
        assert not isinstance(other, LiftedAccessor)

        class Accessor:
            def __getitem__(_, offsets):
                self_offsets = AxisOffset.collect(*self.offsets, *offsets)
                return op(other, self.accessor[self_offsets])

        return LiftedAccessor(accessor=Accessor(), offsets=())

    def __mul__(self, other):
        return self._op(other, operator.mul)

    def __rmul__(self, other):
        return self._rop(other, operator.mul)

    def __add__(self, other):
        return self._op(other, operator.add)

    def __sub__(self, other):
        return self._op(other, operator.sub)

    def __gt__(self, other):
        return self._op(other, operator.gt)


@dataclass(frozen=True)
class Domain:
    i: int
    j: int
    k: int


def stencil(func):
    return func


def fencil(func):
    return func


@dataclass(frozen=True)
class ArrayAccessor:
    array: np.ndarray
    focus: tuple[int, int, int]

    def _set(self, value):
        self.array[self.focus] = value

    def __getitem__(self, offsets):
        if isinstance(offsets, AxisOffset):
            offsets = (offsets,)
        offset_dict = {o.axis: o.offset for o in offsets}
        absolute_index = tuple(
            f + offset_dict.get(axis, 0) for axis, f in enumerate(self.focus)
        )
        assert all(i >= 0 for i in absolute_index)
        return self.array[absolute_index]

    def _shifted(self, *offset):
        return ArrayAccessor(
            array=self.array, focus=tuple(f + i for f, i in zip(self.focus, offset))
        )

    def __array__(self):
        return self.array


def apply_stencil(stencil, domain, outputs, inputs):
    assert len(outputs) == 1
    for i in range(domain.i):
        for j in range(domain.j):
            for k in range(domain.k):
                ijk_inputs = [
                    LiftedAccessor(accessor=inp._shifted(i, j, k), offsets=())
                    for inp in inputs
                ]
                ijk_outputs = [out._shifted(i, j, k) for out in outputs]
                res = stencil(*ijk_inputs)._evaluate()
                ijk_outputs[0]._set(res)


def storage(array, origin=None):
    if origin is None:
        origin = (0,) * array.ndim
    return ArrayAccessor(array=np.copy(array), focus=origin)


def domain(*args):
    if len(args) == 1:
        args = args[0]
    return Domain(i=args[0], j=args[1], k=args[2])
