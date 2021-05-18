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

from unstructured.concepts import element_access_to_field
from unstructured.utils import print_axises


def as_1d(arr):
    return arr.flatten()


def as_2d(arr, shape):
    return arr.reshape(*shape)


def array_as_field(*dims, element_type=None, tuple_size=None):
    def _fun(np_arr):
        assert np_arr.ndim == len(dims)
        for i in range(len(dims)):
            if hasattr(dims[i], "__len__"):
                assert (
                    len(dims[i](0)) == np_arr.shape[i]
                )  # TODO dim[i](0) assumes I can construct the index 0

        @element_access_to_field(
            axises=dims, element_type=element_type, tuple_size=tuple_size
        )
        def element_access(indices):
            def _order_indices(indices):
                lst = []
                types = tuple((type(ind) for ind in indices))
                for axis in dims:
                    lst.append(indices[types.index(axis)].__index__())
                return tuple(lst)

            assert len(indices) == len(dims)
            element = np_arr[_order_indices(indices)]
            return element_type(element) if element_type is not None else element

        return element_access

    return _fun


def constant_field(*dims):
    def _impl(c):
        @element_access_to_field(axises=dims, element_type=type(c), tuple_size=None)
        def _field(_):
            return c

        return _field

    return _impl


def index_field(loc):
    @element_access_to_field(axises=(loc,), element_type=int, tuple_size=None)
    def fun(index):
        assert len(index) == 1
        return index[0].__index__()

    return fun
