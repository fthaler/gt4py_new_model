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

from unstructured.concepts import neighborhood, LocationType
from unstructured.helpers import simple_connectivity


@neighborhood(LocationType.Vertex, LocationType.Vertex)
class CartesianNeighborHood:
    pass


@simple_connectivity(CartesianNeighborHood())
def cartesian_connectivity(*indices):
    class neighs:
        def __getitem__(self, neighindices):
            if not isinstance(neighindices, tuple):
                neighindices = (neighindices,)
            return tuple(
                map(lambda x: x[0] + x[1], zip(indices, neighindices)),
            )

    return neighs()
