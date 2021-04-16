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

from abc import ABC
import enum
import itertools
from typing import Iterable


def ufield(loc):
    def _impl(fun):
        class _field:
            location = loc

            def __call__(self, *indices):
                return fun(*indices)

        return _field()

    return _impl


def connectivity(*_neighborhoods):
    def _impl(fun):
        class conn:
            neighborhoods = _neighborhoods
            in_location, out_location = (
                neighborhoods[0].in_location,
                neighborhoods[-1].out_location,
            )

            def __call__(self, field):
                assert hasattr(field, "location")
                assert field.location == self.out_location

                return fun(field)

        return conn()

    return _impl


def conn_mult(conn_a, conn_b):
    assert conn_a.neighborhoods[-1].out_location == conn_b.neighborhoods[0].in_location

    class conn:
        neighborhoods = [*conn_a.neighborhoods, *conn_b.neighborhoods]
        in_location, out_location = (
            neighborhoods[0].in_location,
            neighborhoods[-1].out_location,
        )

        def __call__(self, field):
            return conn_a(conn_b(field))

    return conn()


def stencil(*args):
    def _impl(fun):
        class decorated_stencil:
            acc_neighborhood_chains = args

            def __call__(self, *args):
                return fun(*args)

        return decorated_stencil()

    return _impl


class Accessor(ABC):
    @classmethod
    def __subclasshook__(cls, C):
        if hasattr(C, "neighborhoods") and hasattr(C, "__getitem__"):
            return True
        return NotImplemented


# wrap a plain accessor with its neighborhoodchain
def accessor(*_neighborhoods):
    def _impl(plain_acc):  # plain_acc is callable
        class acc:
            neighborhoods = _neighborhoods

            def __getitem__(self, *indices):
                return plain_acc(*indices)

        return acc()

    return _impl


def lift(stencil):
    def lifted(*acc):
        class wrap:
            def __getitem__(self, indices):
                if all(map(lambda x: x[indices] is not None, acc)):
                    return stencil(*map(lambda x: x[indices], acc))
                else:
                    return None

        return wrap()

    return lifted


class NeighborHood:
    @classmethod
    def __subclasshook__(cls, C):
        if hasattr(C, "in_location") and hasattr(C, "out_location"):
            return True
        return NotImplemented


def neighborhood(in_loc, out_loc):
    """syntactic sugar to create a neighborhood"""

    def impl(cls):
        cls.in_location = in_loc
        cls.out_location = out_loc
        return cls

    return impl


def get_neighborhood_from_field(field):
    if isinstance(field(0), Accessor):
        return field(0).neighborhoods
    else:
        return ()


def apply_stencil(stencil, domain, connectivities, out, ins):
    def neighborhood_to_key(neighborhood):
        return type(neighborhood).__name__

    conn_map = {}
    for c in connectivities:
        assert (
            len(c.neighborhoods) == 1
        )  # we assume the user passes only non-multiplied connectivities (just for simplification)
        conn_map[neighborhood_to_key(c.neighborhoods[0])] = c

    assert len(stencil.acc_neighborhood_chains) == len(ins)

    stencil_args = []
    for inp, nh_chain in zip(ins, stencil.acc_neighborhood_chains):
        cur_conn = None
        if nh_chain:
            existing_input_neighborhood = get_neighborhood_from_field(inp)
            while nh_chain != existing_input_neighborhood:
                nh, *nh_chain2 = nh_chain  # TODO
                nh_chain = (*nh_chain2,)
                if not cur_conn:
                    cur_conn = conn_map[neighborhood_to_key(nh)]
                else:
                    cur_conn = conn_mult(cur_conn, conn_map[neighborhood_to_key(nh)])
        if cur_conn:
            stencil_args.append(cur_conn(inp))
        else:
            stencil_args.append(inp)

    for indices in itertools.product(*domain):
        res = stencil(*map(lambda fun: fun(*indices), stencil_args))
        if not isinstance(res, tuple):
            res = (res,)

        assert len(res) == len(out)
        for i in range(len(res)):
            out[i][indices] = res[i]


@enum.unique
class LocationType(enum.IntEnum):
    Vertex = 0
    Edge = 1
    Cell = 2
