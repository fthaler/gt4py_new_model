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

import numpy as np
import math

from unstructured.concepts import LocationType, apply_stencil, stencil
from unstructured.helpers import as_field, as_1d, as_2d


@stencil(())
def copy(acc_in):
    return acc_in


def test_copy():
    shape = (5, 7)
    inp = np.random.rand(*shape)
    out1d = np.zeros(math.prod(shape))

    inp1d = as_1d(inp)

    domain = list(range(math.prod(shape)))

    apply_stencil(copy, [domain], [], out1d, [as_field(inp1d, LocationType.Vertex)])
    out2d = as_2d(out1d, shape)
    assert np.allclose(out2d, inp)


test_copy()
