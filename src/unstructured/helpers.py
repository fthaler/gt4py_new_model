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

from unstructured.concepts import Field, field_dec, _tupelize, field_slice, print_axises


def as_1d(arr):
    return arr.flatten()


def as_2d(arr, shape):
    return arr.reshape(*shape)


# def array_to_field(arr, axises):
#     @field_dec(axises)
#     def get_item_to_index(*indices):
#         return arr[indices]

#     return get_item_to_index


def remove_axis(axis, axises):
    i = axises.index(axis)
    return axises[:i] + axises[i + 1 :]


def remove_axises_from_axises(to_remove, axises):
    res = axises
    for axis in to_remove:
        res = remove_axis(axis, res)
    return res


def get_index_of_type(loc):
    def fun(indices):
        for ind in indices:
            if isinstance(ind, loc):
                return ind

    return fun


def make_field(element_access, bind_indices, axises):
    class _sliced_field(Field):
        def __init__(self):
            self.axises = remove_axises_from_axises(
                (type(i) for i in bind_indices), axises
            )

        def __getitem__(self, indices):
            indices = _tupelize(indices)
            if len(indices) == len(self.axises):
                return element_access(bind_indices + indices)
            else:
                return make_field(
                    element_access,
                    indices,
                    self.axises,
                )

    return _sliced_field()


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
