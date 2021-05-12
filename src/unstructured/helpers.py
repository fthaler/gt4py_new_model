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

from unstructured.concepts import Field, _tupelize
from unstructured.utils import remove_axises_from_axises


def as_1d(arr):
    return arr.flatten()


def as_2d(arr, shape):
    return arr.reshape(*shape)


def make_field(element_access, bind_indices, axises):
    axises = _tupelize(axises)

    class _field(Field):
        def __init__(self):
            self.axises = remove_axises_from_axises(
                (type(i) for i in bind_indices), axises
            )

        def __getitem__(self, indices):
            indices = _tupelize(indices)
            if len(indices) == len(self.axises):
                return element_access(bind_indices + indices)
            else:
                # field with `indices` bound
                return make_field(
                    element_access,
                    indices,
                    self.axises,
                )

    return _field()


def element_access_to_field(*, axises):
    def _fun(element_access):
        return make_field(element_access, tuple(), axises)

    return _fun


def array_as_field(*dims):
    def _fun(np_arr):
        assert np_arr.ndim == len(dims)
        for i in range(len(dims)):
            if hasattr(dims[i], "__len__"):
                assert (
                    len(dims[i](0)) == np_arr.shape[i]
                )  # TODO dim[i](0) assumes I can construct the index 0

        @element_access_to_field(axises=dims)
        def element_access(indices):
            def _order_indices(indices):
                lst = []
                types = tuple((type(ind) for ind in indices))
                for axis in dims:
                    lst.append(indices[types.index(axis)].__index__())
                return tuple(lst)

            assert len(indices) == len(dims)
            return np_arr[_order_indices(indices)]

        return element_access

    return _fun


def constant_field(c, loc):
    @element_access_to_field(axises=loc)
    def _field(index):
        return c

    return _field
