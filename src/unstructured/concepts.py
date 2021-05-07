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

from abc import ABC, abstractmethod
import enum
import itertools


class Field:  # (ABC):
    # @abstractmethod
    def __call__(self, index):
        return None

    def __add__(outer_self, other):
        # assert outer_self.loc == other.loc

        class _field(Field):
            def __call__(self, index):
                return outer_self(index) + other(index)

        return _field()

    def __sub__(outer_self, other):
        class _field(Field):
            def __call__(self, index):
                return outer_self(index) - other(index)

        return _field()

    def __mul__(outer_self, other):
        class _field(Field):
            def __call__(self, index):
                return (
                    outer_self(index) * other(index)
                    if outer_self(index) and other(index)
                    else None
                )

        return _field()

    def __truediv__(outer_self, other):
        class _field(Field):
            def __call__(self, index):
                return outer_self(index) / other(index)

        return _field()

    def __rmul__(outer_self, other):
        class _field(Field):
            def __call__(self, index):
                return outer_self(index) * other

        return _field()

    def __gt__(outer_self, other):
        class _field(Field):
            def __call__(self, index):
                other_val = other(index) if isinstance(other, Field) else other
                return outer_self(index) > other_val

        return _field()

    def __or__(outer_self, other):
        class _field(Field):
            def __call__(self, index):
                return outer_self(index) or other(index)

        return _field()

    def __eq__(outer_self, other):
        class _field(Field):
            def __call__(self, index):
                return outer_self(index) == other(index)

        return _field()

    def __ne__(outer_self, other):
        class _field(Field):
            def __call__(self, index):
                return outer_self(index) != other(index)

        return _field()


def field_dec(_loc):
    def inner_field_dec(fun):
        class _field(Field):
            loc = _loc

            def __call__(self, index):
                return fun(index)

        return _field()

    return inner_field_dec


class Accessor:
    def __mul__(outer_self, other):
        class acc(Accessor):
            def __getitem__(self, neighindex):
                return outer_self[neighindex] * other[neighindex]

            def __len__(self):
                return len(outer_self)

        return acc()

    def __or__(outer_self, other):
        class acc(Accessor):
            def __getitem__(self, neighindex):
                return outer_self[neighindex] or other[neighindex]

            def __len__(self):
                return len(outer_self)

        return acc()

    def __eq__(outer_self, other):
        class acc(Accessor):
            def __getitem__(self, neighindex):
                return outer_self[neighindex] == other[neighindex]

            def __len__(self):
                return len(outer_self)

        return acc()

    def __ne__(outer_self, other):
        class acc(Accessor):
            def __getitem__(self, neighindex):
                return outer_self[neighindex] != other[neighindex]

            def __len__(self):
                return len(outer_self)

        return acc()


def if_(cond, true_branch, false_branch):
    if isinstance(cond, Field):
        assert isinstance(true_branch, Field) and isinstance(false_branch, Field)

        class _field(Field):
            def __call__(self, index):
                return true_branch(index) if cond(index) else false_branch(index)

        return _field()
    elif isinstance(cond, Accessor):

        class acc(Accessor):
            def __getitem__(self, neighindex):
                # we can promote a field to an accessor of the field
                cur_true_branch = (
                    true_branch[neighindex]
                    if isinstance(true_branch, Accessor)
                    else true_branch
                )
                cur_false_branch = (
                    false_branch[neighindex]
                    if isinstance(false_branch, Accessor)
                    else false_branch
                )

                return if_(cond[neighindex], cur_true_branch, cur_false_branch)

            def __len__(self):
                return len(cond)

        return acc()
    else:
        assert False


def apply_stencil(stencil, domain, connectivities_and_in_fields, out):
    for indices in itertools.product(*domain):
        fields_and_sparse_fields = stencil(*connectivities_and_in_fields)
        res = fields_and_sparse_fields(
            *indices
        )  # TODO loop over neighbors in case of sparse_field
        if not isinstance(res, tuple):
            res = (res,)

        assert len(res) == len(out)
        for i in range(len(res)):
            out[i][indices] = res[i]


@enum.unique
class LocationType(enum.IntEnum):
    Vertex = 0
    Edge = 1
    Cell = 2
