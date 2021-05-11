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

from unstructured.concepts import Field, field_dec, _tupelize, field_slice


def as_1d(arr):
    return arr.flatten()


def as_2d(arr, shape):
    return arr.reshape(*shape)


# def array_to_field(arr, axises):
#     @field_dec(axises)
#     def get_item_to_index(*indices):
#         return arr[indices]

#     return get_item_to_index


def np_as_field(*dims):
    def _fun(np_arr):
        assert np_arr.ndim == len(dims)
        for i in range(len(dims)):
            if hasattr(dims[i], "__len__"):
                assert (
                    len(dims[i](0)) == np_arr.shape[i]
                )  # TODO dim[i](0) assumes I can construct the index 0

        def _order_indices(indices):
            lst = []
            types = tuple((type(ind) for ind in indices))
            for axis in dims:
                lst.append(indices[types.index(axis)].__index__())
            return tuple(lst)

        class _field(Field):
            def __init__(self):
                self.axises = dims

            def __getitem__(self, indices):
                indices = _tupelize(indices)
                if len(indices) == len(dims):
                    return np_arr[_order_indices(indices)]
                else:
                    return field_slice(indices)(self)

        return _field()

        # @field_dec(dims)
        # def _np_field(*indices):

        # return _np_field

    return _fun
