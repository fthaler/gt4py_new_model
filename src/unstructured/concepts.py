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
from numbers import Number

from unstructured.utils import (
    print_axises,
    tupelize,
    remove_axis,
    remove_axises_from_axises,
    split_indices,
)


def make_field(element_access, bind_indices, axises, element_type):
    axises = tupelize(axises)

    class _field(Field):
        def __init__(self):
            self.element_type = element_type
            self.axises = remove_axises_from_axises(
                (type(i) for i in bind_indices), axises
            )

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

    return _field()


def element_access_to_field(*, axises, element_type):
    def _fun(element_access):
        return make_field(element_access, tuple(), axises, element_type)

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
    )
    def element_access(indices):
        index_field_indices, rest = split_indices(indices, index_field.axises)
        return field[(index_field[index_field_indices],) + rest]

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
    SUPPORTED_REVERSE_OPS = ["__rmul__"]

    @staticmethod
    def _field_op(op):
        def fun(first, second):
            if isinstance(second, Field):
                assert first.axises == second.axises  # TODO order independant
                print(first.element_type)
                print(second.element_type)
                if first.element_type is not None and second.element_type is not None:
                    assert first.element_type == second.element_type

            class _field(Field):
                axises = first.axises
                element_type = first.element_type

                def __getitem__(self, index):
                    if isinstance(second, Field):
                        second_value = second[index]
                    elif isinstance(second, Number):
                        second_value = second
                    else:
                        raise ValueError()

                    if first[index] is not None:
                        assert second_value is not None
                        return op(first[index], second_value)
                    else:
                        return None

            if hasattr(first, "__len__") and hasattr(second, "__len__"):
                assert len(first) == len(second)
                setattr(_field, "__len__", first.__len__)
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


def field_dec(axises, *, element_type=None):
    axises = tupelize(axises)

    def inner_field_dec(fun):
        class _field(Field):
            def __init__(self):
                self.axises = axises
                self.element_type = element_type

            def __getitem__(self, index):
                index = tupelize(index)
                return fun(index)

        return _field()

    return inner_field_dec


def if_(cond, true_branch, false_branch):
    assert isinstance(cond, Field)
    assert isinstance(true_branch, Field)
    assert isinstance(false_branch, Field)

    @field_dec(true_branch.axises)
    def _field(index):
        return true_branch[index] if cond[index] else false_branch[index]

    return _field


def reduce(op, init):
    def _red_dim(dim):
        if not hasattr(dim(0), "__len__"):
            raise TypeError("Dimension is not reducible")

        def _reduce(field):
            class _ReducedField:
                axises = tuple(axis for axis in field.axises if not axis is dim)

                def __getitem__(self, indices):
                    indices = tupelize(indices)
                    res = init
                    for i in range(len(dim(0))):
                        res = op(res, field[(dim(i),) + indices])
                    return res

            return _ReducedField()

        return _reduce

    return _red_dim


def sum_reduce(dim):
    return reduce(operator.add, 0)(dim)


def apply_stencil(stencil, domain, connectivities_and_in_fields, out):
    assert len(domain) == 1  # TODO
    indices, ind_type = domain[0]
    for index in indices:
        fields_and_sparse_fields = stencil(*connectivities_and_in_fields)
        res = fields_and_sparse_fields[
            ind_type(index)
        ]  # TODO loop over neighbors in case of sparse_field
        if not isinstance(res, tuple):
            res = (res,)

        assert len(res) == len(out)
        for i in range(len(res)):
            out[i][index] = res[i]
