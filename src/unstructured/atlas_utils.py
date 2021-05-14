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

from unstructured.utils import get_index_of_type
from unstructured.concepts import element_access_to_field
from atlas4py import IrregularConnectivity


def make_sparse_index_field_from_atlas_connectivity(
    atlas_connectivity, primary_loc, neigh_loc, field_loc
):
    @element_access_to_field(axises=(primary_loc, neigh_loc), element_type=field_loc)
    def element_access(indices):
        primary_index = get_index_of_type(primary_loc)(indices)
        neigh_index = get_index_of_type(neigh_loc)(indices)
        if isinstance(atlas_connectivity, IrregularConnectivity):
            if neigh_index.__index__() < atlas_connectivity.cols(
                primary_index.__index__()
            ):
                return field_loc(
                    atlas_connectivity[
                        primary_index.__index__(), neigh_index.__index__()
                    ]
                )
            else:
                return None
        else:
            if neigh_index.__index__() < 2:
                return field_loc(
                    atlas_connectivity[
                        primary_index.__index__(), neigh_index.__index__()
                    ]
                )
            else:
                assert False

    return element_access
