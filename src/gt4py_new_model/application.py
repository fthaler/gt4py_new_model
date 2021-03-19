from dataclasses import dataclass
from numbers import Number
from typing import Optional

from .dimensions import Dimension
from .types import Accessor, Column, Stencil


@dataclass(frozen=True)
class Domain:
    i: int
    j: int
    k: int


def domain(*args):
    if len(args) == 1:
        args = args[0]
    return Domain(i=args[0], j=args[1], k=args[2])


def stencil(func: Stencil) -> Stencil:
    return func


@dataclass(frozen=True)
class LiftedStencilAccessor:
    stencil: Stencil
    args: tuple[Accessor, ...]
    tuple_index: Optional[int] = None

    def __getitem__(self, offsets):
        if not isinstance(offsets, tuple):
            offsets = (offsets,)

        @dataclass(frozen=True)
        class WrappedAccessor:
            accessor: Accessor

            def __getitem__(acc, offs):
                if not isinstance(offs, tuple):
                    offs = (offs,)
                return acc.accessor[Dimension.collect(*offsets, *offs)]

            def __getattr__(acc, attr):
                if not hasattr(acc.accessor, attr):
                    return AttributeError()
                return getattr(acc.accessor, attr)

        def wrap(arg):
            if isinstance(arg, (Number, tuple, Dimension)) or callable(arg):
                return arg
            return WrappedAccessor(accessor=arg)

        res = self.stencil(*(wrap(arg) for arg in self.args))
        if self.tuple_index is not None:
            return res[self.tuple_index]
        return res

    def _value_accessor(self, k):
        class ValueAccessor:
            def __getitem__(acc, offsets):
                return self[offsets][k]

        return ValueAccessor()


def lift(stencil, return_values=1):
    def lifted(*args):
        if return_values > 1:

            class Wrapper:
                def __getitem__(self, tuple_index):
                    if isinstance(tuple_index, int):
                        if tuple_index < 0 or tuple_index >= len(self):
                            raise IndexError()
                        return LiftedStencilAccessor(
                            stencil=stencil, args=args, tuple_index=tuple_index
                        )
                    assert isinstance(tuple_index, slice)
                    return [self[i] for i in range(len(self))]

                def __len__(self):
                    return return_values

            return Wrapper()

        return LiftedStencilAccessor(stencil=stencil, args=args)

    return lifted


def fencil(func):
    return func


def scaniter(func):
    return func


def apply_stencil(stencil, domain, outputs, inputs):
    def setval(storage, i: int, j: int, k: int, value: Column):
        index = tuple(
            o + i if d == "i" else o + j if d == "j" else o + k if d == "k" else o
            for o, d in zip(storage.origin, storage.dimensions)
        )
        storage.array[index] = value[k]

    for i in range(domain.i):
        for j in range(domain.j):
            in_accessors = (inp._k_column_accessor(i, j, domain.k) for inp in inputs)
            res = stencil(*in_accessors)
            if not isinstance(res, tuple):
                res = (res,)
            for out, r in zip(outputs, res):
                for k in range(domain.k):
                    setval(out, i, j, k, r)
