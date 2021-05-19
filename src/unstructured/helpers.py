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
from unstructured.utils import Dimension, make_dimensions, split_indices
import numpy as np


def as_1d(arr):
    return arr.flatten()


def as_2d(arr, shape):
    return arr.reshape(*shape)


def field_sequence_as_field(dim):
    def _fun(seq):
        # TODO assert all elements have same axises
        @element_access_to_field(
            axises=(dim,) + seq[0].axises,
            element_type=seq[0].element_type,
            tuple_size=seq[0].tuple_size,
        )
        def elem_acc(indices):
            current_dim, rest = split_indices(indices, (dim,))
            assert len(current_dim) == 1
            current_dim = current_dim[0]
            return seq[current_dim][rest]

        return elem_acc

    return _fun


def array_as_field(*axises, element_type=None, tuple_size=None):
    def _fun(np_arr):
        assert np_arr.ndim == len(axises)
        for i in range(len(axises)):
            if hasattr(axises[i], "__len__"):
                assert (
                    len(axises[i](0)) == np_arr.shape[i]
                )  # TODO dim[i](0) assumes I can construct the index 0

        @element_access_to_field(
            dimensions=make_dimensions(axises, tuple(range(i) for i in np_arr.shape)),
            element_type=element_type,
            tuple_size=tuple_size,
        )
        def element_access(indices):
            def _order_indices(indices):
                lst = []
                types = tuple((type(ind) for ind in indices))
                for axis in axises:
                    lst.append(indices[types.index(axis)].__index__())
                return tuple(lst)

            element = np_arr[_order_indices(indices)]
            return element_type(element) if element_type is not None else element

        return element_access

    return _fun


def materialize(field):
    return array_as_field(
        *(dim.axis for dim in field.dimensions), element_type=field.element_type
    )(np.asarray(field))


def constant_field(*axises):
    def _impl(c):
        @element_access_to_field(
            dimensions=tuple(Dimension(axis, None) for axis in axises),
            element_type=type(c),
            tuple_size=None,
        )
        def _field(_):
            return c

        return _field

    return _impl


def index_field(loc, range=None):
    @element_access_to_field(
        dimensions=Dimension(loc, range), element_type=int, tuple_size=None
    )
    def fun(index):
        assert len(index) == 1
        return index[0].__index__()

    return fun
