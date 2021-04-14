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

from importlib.metadata import version

__version__ = version(__package__)

from .dimensions import I, J, K
from .application import apply_stencil, domain, fencil, lift, liftv, stencil
from .builtins import forward, backward, polymorphic_stencil
from .storage import storage, index, constant

__all__ = [
    "I",
    "J",
    "K",
    "apply_stencil",
    "domain",
    "fencil",
    "lift",
    "liftv",
    "stencil",
    "polymorphic_stencil",
    "storage",
    "index",
    "constant",
    "forward",
    "backward",
]
