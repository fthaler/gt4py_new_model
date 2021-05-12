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

import enum
import itertools
import operator
from numbers import Number

from unstructured.utils import _tupelize


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

            class _field(Field):
                axises = first.axises

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


def field_dec(axises):
    axises = _tupelize(axises)

    def inner_field_dec(fun):
        class _field(Field):
            def __init__(self):
                self.axises = axises

            def __getitem__(self, index):
                index = _tupelize(index)
                return fun(index)

        return _field()

    return inner_field_dec


def axis(*, length=None, aliases=None):
    def _impl(cls):
        class _axis:
            def __init__(self, index):
                self.index = index

            def __index__(self):
                if length is not None:
                    if self.index >= length:
                        raise IndexError()

                return self.index

            def __str__(self):
                return f"{cls.__name__}({self.index})"

            def __eq__(self, other):
                return type(self) == type(other) and self.index == other.__index__()

        if length is not None:
            setattr(_axis, "__len__", lambda self: length)

        if aliases:
            for i, alias in enumerate(aliases):
                setattr(_axis, alias, _axis(i))

        return _axis

    return _impl


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
                    indices = _tupelize(indices)
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
