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

from unstructured.concepts import Accessor, field_dec
import numpy as np

# Location tags
Nodes = 0
Links = 1


def one_dimensional_neighbor_connectivity(field):
    assert field.loc == Links

    class acc(Accessor):
        def __len__(self):
            return 2

        def __getitem__(self, neighindex):
            @field_dec(Nodes)
            def _field(index):
                if neighindex < 0:
                    neighindex = neighindex + 1
                return field(index + neighindex)

            return _field

    return acc()


def sparse_field_stencil(conn, inp):
    return conn(inp)


def test_sparse_field():
    n_nodes = 10
    n_links = 9

    out = np.zeros((n_nodes))
    inp

    sparse_field_stencil(one_dimensional_neighbor_connectivity, inp)[0]


test_sparse_field()
