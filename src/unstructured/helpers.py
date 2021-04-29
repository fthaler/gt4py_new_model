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

from unstructured.concepts import LocationType, accessor, connectivity, ufield


def as_1d(arr):
    return arr.flatten()


def as_2d(arr, shape):
    return arr.reshape(*shape)


# a field is a function from index to element `()` not `[]`
# (or change the conn)
def as_field(arr, loc: LocationType):
    class _field:
        location = loc

        def __call__(self, *indices):
            return arr[indices]

    return _field()


def simple_connectivity(neighborhood):
    def _impl(fun):  # fun is function from index to array of neighbor index
        @connectivity(neighborhood)
        def conn(field):
            @ufield(neighborhood.in_location)
            def _field(*index):
                @accessor(neighborhood)
                def neighs(indices):
                    res = fun(*index)[indices]
                    if not isinstance(res, tuple):
                        res = (res,)
                    if all(map(lambda x: x is not None, res)):
                        return field(*res)
                    else:
                        # neighbor doesn't exist
                        return None

                return neighs

            return _field

        return conn

    return _impl
