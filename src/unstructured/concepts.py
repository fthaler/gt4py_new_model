# GT4Py New Semantic Model - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.  GT4Py
# New Semantic Model is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or any later version.
# See the LICENSE.txt file at the top-level directory of this distribution for
# a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import operator
import itertools
from numbers import Number

from unstructured.utils import (
    axis,
    print_axises,
    remove_indices_of_axises,
    tupelize,
    remove_axis,
    remove_axises_from_axises,
    split_indices,
)


@axis()
class TupleDim__:
    """Builtin axis that can be unpacked"""

    ...


# tuple size is hacked on top...
def make_field(element_access, bind_indices, axises, element_type, *, tuple_size=None):
    axises = tupelize(axises)

    class _field(Field):
        def __init__(self):
            self.element_type = element_type
            self.axises = remove_axises_from_axises(
                (type(i) for i in bind_indices), axises
            )
            self.tuple_size = tuple_size

        def __getitem__(self, indices):
            if isinstance(indices, Field):
                return field_getitem(self, indices)

            else:
                indices = tupelize(indices)
                if len(indices) == len(self.axises):
                    return element_access(bind_indices + indices)
                else:
                    # field with `indices` bound
                    return make_field(
                        element_access, indices, self.axises, element_type
                    )

        def __iter__(self):
            assert TupleDim__ in self.axises
            assert self.tuple_size is not None

            def make_tuple_acc(i):
                @element_access_to_field(
                    axises=remove_axis(TupleDim__, self.axises),
                    element_type=self.element_type,
                    tuple_size=0,
                )
                def tuple_acc(indices):
                    return self[TupleDim__(i)][indices]

                return tuple_acc

            return iter(
                map(
                    lambda i: make_tuple_acc(i),
                    range(self.tuple_size),
                )
            )

    return _field()


def element_access_to_field(*, axises, element_type, tuple_size):
    def _fun(element_access):
        return make_field(
            element_access, tuple(), axises, element_type, tuple_size=tuple_size
        )

    return _fun


def field_getitem(field, index_field):
    assert index_field.element_type is not None
    if index_field.element_type is None:
        raise TypeError("Is not an index field, missing element type.")
    if not index_field.element_type in field.axises:
        raise TypeError("Incompatible index field passed.")

    @element_access_to_field(
        axises=(
            remove_axis(index_field.element_type, field.axises) + index_field.axises
        ),
        element_type=field.element_type,
        tuple_size=field.tuple_size,
    )
    def element_access(indices):
        index_field_indices, rest = split_indices(indices, index_field.axises)
        new_index = index_field[index_field_indices]
        if new_index is not None:
            return field[(new_index,) + rest]
        else:
            return None

    return element_access


class _FieldArithmetic:
    SUPPORTED_OPS = [
        "__mul__",
        "__add__",
        "__sub__",
        "__truediv__",
        "__gt__",
        "__lt__",
        "__or__",
        "__and__",
        "__eq__",
        "__ne__",
    ]
    SUPPORTED_REVERSE_OPS = ["__rmul__", "__radd__"]

    @staticmethod
    def _field_op(op):
        def fun(first, second):
            if isinstance(second, Field):
                assert first.axises == second.axises  # TODO order independant
                if first.element_type is not None and second.element_type is not None:
                    assert first.element_type == second.element_type

            class _field(Field):
                axises = first.axises
                element_type = first.element_type
                tuple_size = first.tuple_size

                def __getitem__(self, index):
                    if isinstance(second, Field):
                        second_value = second[index]
                    elif isinstance(second, Number):
                        second_value = second
                    else:
                        raise ValueError()

                    if first[index] is not None and second_value is not None:
                        return op(first[index], second_value)
                    else:
                        return None

            # if hasattr(first, "__len__") and hasattr(second, "__len__"):
            #     assert len(first) == len(second)
            #     setattr(_field, "__len__", first.__len__)
            return _field()

        return fun

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        attr_to_op = {op: op for op in cls.SUPPORTED_OPS}
        attr_to_op |= {op: op.replace("__r", "__") for op in cls.SUPPORTED_REVERSE_OPS}
        for attr, op in attr_to_op.items():
            setattr(
                cls,
                attr,
                lambda self, other, op=op: self._field_op(getattr(operator, op))(
                    self, other
                ),
            )


class Field(_FieldArithmetic):
    pass


def if_(cond, true_branch, false_branch):
    assert isinstance(cond, Field)
    assert isinstance(true_branch, Field)
    assert isinstance(false_branch, Field)

    assert true_branch.axises == false_branch.axises
    assert cond.axises == true_branch.axises
    if true_branch.element_type is not None and false_branch.element_type is not None:
        assert true_branch.element_type == false_branch.element_type
        element_type = true_branch.element_type
    else:
        element_type = None

    assert cond.tuple_size == true_branch.tuple_size
    assert true_branch.tuple_size == false_branch.tuple_size

    @element_access_to_field(
        axises=true_branch.axises,
        element_type=element_type,
        tuple_size=true_branch.tuple_size,
    )
    def elem_acc(indices):
        return true_branch[indices] if cond[indices] else false_branch[indices]

    return elem_acc


def reduce(op, init):
    def _red_dim(dim):
        if not hasattr(dim(0), "__len__"):
            raise TypeError("Dimension is not reducible")

        def _reduce(field):
            @element_access_to_field(
                axises=remove_axis(dim, field.axises),
                element_type=field.element_type,
                tuple_size=field.tuple_size,
            )
            def elem_access(indices):
                indices = tupelize(indices)
                res = init
                for i in range(len(dim(0))):
                    val = field[(dim(i),) + indices]
                    if val is not None:
                        res = op(res, val)
                return res

            return elem_access

        return _reduce

    return _red_dim


def sum_reduce(dim):
    return reduce(operator.add, 0)(dim)


def broadcast(*dims):
    def _impl(field):
        @element_access_to_field(
            axises=field.axises + dims,
            element_type=field.element_type,
            tuple_size=field.tuple_size,
        )
        def elem_access(indices):
            return field[remove_indices_of_axises(dims, indices)]

        return elem_access

    return _impl


def apply_stencil(stencil, domain, connectivities_and_in_fields, out):
    ranges, types = tuple(zip(*domain))

    for indices in itertools.product(*ranges):
        fields = tupelize(stencil(*connectivities_and_in_fields))
        assert len(fields) == len(out)

        for o, f in zip(out, fields):
            typed_indices = tuple(map(lambda i_t: i_t[1](i_t[0]), zip(indices, types)))
            o[indices] = f[typed_indices]


def generic_scan(axis, fun, *, backward):
    def scanner(*inps):
        # TODO assert all inp have the same axises
        axises = inps[0].axises

        def make_elem_access(tuple_size):
            new_axises = axises if tuple_size is None else axises + (TupleDim__,)

            @element_access_to_field(
                axises=new_axises,
                element_type=inps[0].element_type,
                tuple_size=tuple_size,
            )
            def elem_acc(indices):
                tmp_axises = new_axises
                scan_index, rest = split_indices(indices, (axis,))
                assert len(scan_index) == 1

                state = None

                if not backward:
                    iter = list(range(scan_index[0].__index__() + 1))
                else:
                    iter = list(range(scan_index[0], len(axis(0))))
                    iter.reverse()
                    print(iter)
                for ind in map(lambda i: axis(i), iter):
                    state = fun(state, *tuple(inp[ind] for inp in inps))

                if tuple_size is None:
                    return state[rest]
                else:
                    tuple_index, rest = split_indices(rest, (TupleDim__,))
                    assert len(tuple_index) == 1
                    tuple_index = tuple_index[0]
                    return state[tuple_index.__index__()][rest]

            return elem_acc

        # try what the result of fun is (tuple or value):
        res_check = fun(None, *tuple(inp[axis(0)] for inp in inps))
        return make_elem_access(
            None if not isinstance(res_check, tuple) else len(res_check)
        )

    return scanner


def scan_pass(axis, *, backward=False):
    def impl(fun):
        return generic_scan(axis, fun, backward=backward)

    return impl


def forward_scan(axis):
    return scan_pass(axis, backward=False)


def backward_scan(axis):
    return scan_pass(axis, backward=True)
