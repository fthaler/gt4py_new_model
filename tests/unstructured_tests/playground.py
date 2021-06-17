# MDIterator
#  - has `deref()`
#  - has `position`, dict of key (`axis`) to index

# shift(Iterator, GeneralizedOffset) -> Iterator

# GeneralizedOffset is a callable that takes pos and returns new pos


from .fvm_nabla_setup import assert_close, nabla_setup
from typing import Callable, Dict
import numpy as np
import pytest
import itertools

from unstructured.utils import tupelize

from .hdiff_reference import hdiff_reference

# concepts


def get_order_indices(axises, pos):
    return tuple(slice(None) if axis is ExtraDim else pos[axis] for axis in axises)


class MDIterator:
    def __init__(self, field, pos, *, offsets=[]) -> None:
        self.field = field
        self.pos = pos
        self.offsets = offsets

    def shift(self, offset):
        return MDIterator(self.field, self.pos, offsets=[*self.offsets, offset])

    def is_none(self):
        shifted_pos = self.pos
        if shifted_pos is None:
            return True
        for offset in self.offsets:
            shifted_pos = offset(shifted_pos)
            if shifted_pos is None:
                return True
        return False

    def deref(self):
        shifted_pos = self.pos
        if shifted_pos is None:
            return None
        for offset in self.offsets:
            shifted_pos = offset(shifted_pos)
            if shifted_pos is None:
                return None

        if not all(
            axis in [*shifted_pos.keys(), ExtraDim] for axis in self.field.axises
        ):
            raise IndexError(
                "Iterator position doesn't point to valid location for its field."
            )
        ordered_indices = get_order_indices(self.field.axises, shifted_pos)
        return self.field[ordered_indices]


class LocatedField:
    def __init__(self, getter, axises, *, setter=None, array=None):
        self.getter = getter
        self.axises = axises
        self.setter = setter
        self.array = array

    def __getitem__(self, indices):
        indices = tupelize(indices)
        return self.getter(indices)

    def __setitem__(self, indices, value):
        if self.setter is None:
            raise TypeError("__setitem__ not supported for this field")
        self.setter(indices, value)

    def __array__(self):
        if self.array is None:
            raise TypeError("__array__ not supported for this field")
        return self.array()


def shift(iter, offset):  # shift is lazy
    return iter.shift(offset)


def deref(iter):
    return iter.deref()


def lift(stencil):
    def impl(*args):
        class wrap_iterator:
            def __init__(self, *, offsets=[]) -> None:
                self.offsets = offsets

            def shift(self, offset):
                return wrap_iterator(offsets=[*self.offsets, offset])

            def deref(self):
                shifted_args = args
                for offset in self.offsets:
                    shifted_args = tuple(
                        map(lambda arg: shift(arg, offset), shifted_args)
                    )

                if any(shifted_arg.is_none() for shifted_arg in shifted_args):
                    return None
                return stencil(*shifted_args)

        return wrap_iterator()

    return impl


def named_range(axis, range):
    return ((axis, i) for i in range)


def domain_iterator(domain):
    return (
        dict(elem)
        for elem in itertools.product(
            *map(lambda tup: named_range(tup[0], tup[1]), domain.items())
        )
    )


def apply_stencil(sten, domain, ins, outs):  # domain is Dict[axis, range]
    for pos in domain_iterator(domain):
        ins_iters = list(MDIterator(inp, pos) for inp in ins)
        res = sten(*ins_iters)
        if not isinstance(res, tuple):
            res = (res,)
        if not len(res) == len(outs):
            IndexError("Number of return values doesn't match number of output fields.")

        for r, out in zip(res, outs):
            ordered_indices = tuple(get_order_indices(out.axises, pos))
            out[ordered_indices] = r


# this is a hack for having sparse field, we can just pass extra dimensions and you get an array
# this is probably not what we want
class ExtraDim:
    ...


# helpers


def _tupsum(a, b):
    def sum_if_not_slice(ab_elems):
        return (
            slice(None)
            if (isinstance(ab_elems[0], slice) or isinstance(ab_elems[1], slice))
            else sum(ab_elems)
        )

    return tuple(sum_if_not_slice(i) for i in zip(a, b))


def np_as_located_field(*axises, origin=None):
    def _maker(a: np.ndarray):
        if a.ndim != len(axises):
            raise TypeError("ndarray.ndim incompatible with number of given axises")

        if origin is not None:
            offsets = get_order_indices(axises, origin)
        else:
            offsets = tuple(0 for _ in axises)

        def setter(indices, value):
            a[_tupsum(indices, offsets)] = value

        def getter(indices):
            return a[_tupsum(indices, offsets)]

        return LocatedField(getter, axises, setter=setter, array=a.__array__)

    return _maker


# tests
def test_domain():
    class I:
        ...

    class J:
        ...

    domain = {I: range(2), J: range(3)}
    for pos in domain_iterator(domain):
        print(pos)


test_domain()

# user code


class CartesianAxis:
    def __init__(self, offset) -> None:
        self.offset = offset

    def __call__(self, pos: Dict) -> Dict:
        axis = type(self)
        if axis in pos.keys():
            new_pos = pos.copy()
            new_pos[axis] += self.offset
            return new_pos
        return pos


class I(CartesianAxis):
    ...


class J(CartesianAxis):
    ...


class K(CartesianAxis):
    ...


def test_iterator():
    inp = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    inp_located = np_as_located_field(I, J)(inp)

    inp_located[2, 3] = 20
    assert inp_located[2, 3] == 20

    inp_it = MDIterator(inp_located, {I: 2, J: 0})
    print(deref(inp_it))
    assert deref(MDIterator(inp_located, {I: 2, J: 0})) == deref(
        MDIterator(inp_located, {J: 0, I: 2})
    )

    with pytest.raises(IndexError):
        deref(MDIterator(inp_located, {I: 0, K: 0}))


test_iterator()


def test_shift():
    inp = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    inp_located = np_as_located_field(I, J)(inp)
    inp_it = MDIterator(inp_located, {I: 1, J: 2})

    res = shift(shift(inp_it, I(1)), J(-1))
    assert deref(res) == 10


test_shift()


class IStag(CartesianAxis):
    ...


class JStag(CartesianAxis):
    ...


class I2IStag:
    def __init__(self, offset) -> None:
        self.offset = offset

    def __call__(self, pos: Dict) -> Dict:
        if I in pos.keys():
            new_pos = pos.copy()
            ipos = int(new_pos[I] + self.offset - 0.5)
            del new_pos[I]
            new_pos[IStag] = ipos
            return new_pos
        return pos


class IStag2I:
    def __init__(self, offset) -> None:
        self.offset = offset

    def __call__(self, pos: Dict) -> Dict:
        if IStag in pos.keys():
            new_pos = pos.copy()
            ipos = int(new_pos[IStag] + self.offset + 0.5)
            del new_pos[IStag]
            new_pos[I] = ipos
            return new_pos
        return pos


class J2JStag:
    def __init__(self, offset) -> None:
        self.offset = offset

    def __call__(self, pos: Dict) -> Dict:
        if J in pos.keys():
            new_pos = pos.copy()
            ipos = int(new_pos[J] + self.offset - 0.5)
            del new_pos[J]
            new_pos[JStag] = ipos
            return new_pos
        return pos


class JStag2J:
    def __init__(self, offset) -> None:
        self.offset = offset

    def __call__(self, pos: Dict) -> Dict:
        if JStag in pos.keys():
            new_pos = pos.copy()
            ipos = int(new_pos[JStag] + self.offset + 0.5)
            del new_pos[JStag]
            new_pos[J] = ipos
            return new_pos
        return pos


def stag_sum(inp_stag, inp):
    return (
        deref(inp_stag)
        + deref(shift(inp, IStag2I(-0.5)))
        + deref(shift(inp, IStag2I(0.5)))
    )


def broken_stag(inp):
    return deref(inp)


def test_stag():
    inp = np_as_located_field(I)(np.asarray([1, 2, 3, 4, 5]))
    inp_stag = np_as_located_field(IStag)(np.asarray([10, 20, 30, 40]))
    res = np.asarray([13, 25, 37, 49])

    out = np_as_located_field(IStag)(np.zeros([4]))

    # this is the apply stencil explicit
    inp_it = MDIterator(inp, {IStag: 0})
    inp_stag_it = MDIterator(inp_stag, {IStag: 0})
    print(stag_sum(inp_stag_it, inp_it))

    apply_stencil(stag_sum, {IStag: range(4)}, [inp_stag, inp], [out])
    print(out[0])
    print(out[1])
    print(out[2])
    print(out[3])

    with pytest.raises(IndexError):
        apply_stencil(broken_stag, {IStag: range(4)}, [inp], [out])


test_stag()


def lr_sum(inp):
    return deref(shift(inp, I(-1))) + deref(shift(inp, I(1)))


def test_non_zero_origin():
    inp = np_as_located_field(I, origin={I: 1})(np.asarray([1, 2, 3, 4, 5]))
    out = np_as_located_field(I)(np.zeros([3]))
    apply_stencil(lr_sum, {I: range(3)}, [inp], [out])
    assert out[0] == 4
    assert out[1] == 6
    assert out[2] == 8


test_non_zero_origin()


def explicit_nested_sum(inp):
    class shiftable_sum:
        def __init__(self, *, offsets=[]) -> None:
            self.offsets = offsets

        def deref(self):
            shifted_inp = inp
            for offset in self.offsets:
                shifted_inp = shift(shifted_inp, offset)
            return deref(shift(shifted_inp, I(-1))) + deref(shift(shifted_inp, I(1)))

        def shift(self, offset):
            return shiftable_sum(offsets=[*self.offsets, offset])

    return deref(shift(shiftable_sum(), I(1)))


def test_explicit_nested_sum():
    inp = np_as_located_field(I, origin={I: 2})(np.asarray([1, 2, 3, 4, 5]))
    out = np_as_located_field(I)(np.zeros([1]))
    apply_stencil(explicit_nested_sum, {I: range(1)}, [inp], [out])
    print(out[0])


test_explicit_nested_sum()


def nested_sum(inp):
    sum = lift(lr_sum)(inp)
    return lr_sum(sum)


def test_lift():
    inp = np_as_located_field(I, origin={I: 2})(np.asarray([1, 2, 3, 4, 5]))
    out = np_as_located_field(I)(np.zeros([1]))
    apply_stencil(nested_sum, {I: range(1)}, [inp], [out])
    print(out[0])
    assert out[0] == 12


test_lift()


def laplacian(inp):
    return -4 * deref(inp) + (
        deref(shift(inp, I(1)))
        + deref(shift(inp, I(-1)))
        + deref(shift(inp, J(1)))
        + deref(shift(inp, J(-1)))
    )


def hdiff_flux_x(inp):
    lap = lift(laplacian)(inp)
    flux = deref(shift(lap, IStag2I(-0.5))) - deref(shift(lap, IStag2I(0.5)))

    if flux * (deref(shift(inp, IStag2I(0.5))) - deref(shift(inp, IStag2I(-0.5)))) > 0:
        return 0
    else:
        return flux


def hdiff_flux_y(inp):
    lap = lift(laplacian)(inp)
    flux = deref(shift(lap, JStag2J(-0.5))) - deref(shift(lap, JStag2J(0.5)))

    if flux * (deref(shift(inp, JStag2J(0.5))) - deref(shift(inp, JStag2J(-0.5)))) > 0:
        return 0
    else:
        return flux


def hdiff(inp, coeff):
    flx = lift(hdiff_flux_x)(inp)
    fly = lift(hdiff_flux_y)(inp)
    return deref(inp) - deref(coeff) * (
        deref(shift(flx, I2IStag(0.5)))
        - deref(shift(flx, I2IStag(-0.5)))
        + deref(shift(fly, J2JStag(0.5)))
        - deref(shift(fly, J2JStag(-0.5)))
    )


def test_hdiff(hdiff_reference):
    inp, coeff, out = hdiff_reference
    shape = out.shape

    inp_s = np_as_located_field(I, J, K, origin={I: 2, J: 2, K: 0})(inp)
    coeff_s = np_as_located_field(I, J, K)(coeff)
    out_s = np_as_located_field(I, J, K)(np.zeros_like(out))

    domain = {I: range(shape[0]), J: range(shape[1]), K: range(shape[2])}

    apply_stencil(hdiff, domain, [inp_s, coeff_s], [out_s])

    assert np.allclose(out, np.asarray(out_s))


### nabla

fvm_nabla_setup = nabla_setup()


class Vertex:
    ...


class Edge:
    ...


class E2V:
    def __init__(self, neigh_index) -> None:
        self.neigh_index = neigh_index

    def __call__(self, pos: Dict) -> Dict:
        e2v = fvm_nabla_setup.edges2node_connectivity
        if Edge in pos.keys():
            new_pos = pos.copy()
            new_pos[Vertex] = e2v[new_pos[Edge], self.neigh_index]
            del new_pos[Edge]
            return new_pos
        return pos


class V2E:
    def __init__(self, neigh_index) -> None:
        self.neigh_index = neigh_index

    def __call__(self, pos: Dict) -> Dict:
        v2e = fvm_nabla_setup.nodes2edge_connectivity
        if Vertex in pos.keys():
            if self.neigh_index < v2e.cols(pos[Vertex]):
                new_pos = pos.copy()
                new_pos[Edge] = v2e[new_pos[Vertex], self.neigh_index]
                del new_pos[Vertex]
                return new_pos
            else:
                return None
        return pos


def compute_zavgS(pp, S_M):
    zavg = 0.5 * (deref(shift(pp, E2V(0))) + deref(shift(pp, E2V(1))))
    return deref(S_M) * zavg


def compute_pnabla(pp, S_M, sign, vol):
    zavgS = lift(compute_zavgS)(pp, S_M)
    # pnabla_M = sum_reduce(V2E)(zavgS * sign)
    pnabla_M = 0
    for n in range(7):
        zavgS_n = deref(shift(zavgS, V2E(n)))
        if zavgS_n is not None:
            pnabla_M += zavgS_n * deref(sign)[n]

    return pnabla_M / deref(vol)


def nabla(
    pp,
    S_MXX,
    S_MYY,
    sign,
    vol,
):
    return compute_pnabla(pp, S_MXX, sign, vol), compute_pnabla(pp, S_MYY, sign, vol)


def test_compute_zavgS():
    setup = nabla_setup()

    pp = np_as_located_field(Vertex)(setup.input_field)
    S_MXX, S_MYY = tuple(map(np_as_located_field(Edge), setup.S_fields))

    zavgS = np_as_located_field(Edge)(np.zeros((setup.edges_size)))

    apply_stencil(compute_zavgS, {Edge: range(setup.edges_size)}, [pp, S_MXX], [zavgS])
    assert_close(-199755464.25741270, min(zavgS))
    assert_close(388241977.58389181, max(zavgS))

    apply_stencil(compute_zavgS, {Edge: range(setup.edges_size)}, [pp, S_MYY], [zavgS])
    assert_close(-1000788897.3202186, min(zavgS))
    assert_close(1000788897.3202186, max(zavgS))


def test_nabla():
    setup = nabla_setup()

    sign = np_as_located_field(Vertex, ExtraDim)(setup.sign_field)
    pp = np_as_located_field(Vertex)(setup.input_field)
    S_MXX, S_MYY = tuple(map(np_as_located_field(Edge), setup.S_fields))
    vol = np_as_located_field(Vertex)(setup.vol_field)

    pnabla_MXX = np_as_located_field(Vertex)(np.zeros((setup.nodes_size)))
    pnabla_MYY = np_as_located_field(Vertex)(np.zeros((setup.nodes_size)))

    print(f"nodes: {setup.nodes_size}")
    print(f"edges: {setup.edges_size}")

    apply_stencil(
        nabla,
        {Vertex: range(setup.nodes_size)},
        [pp, S_MXX, S_MYY, sign, vol],
        [pnabla_MXX, pnabla_MYY],
    )

    assert_close(-3.5455427772566003e-003, min(pnabla_MXX))
    assert_close(3.5455427772565435e-003, max(pnabla_MXX))
    assert_close(-3.3540113705465301e-003, min(pnabla_MYY))
    assert_close(3.3540113705465301e-003, max(pnabla_MYY))


### indirect vs direct addressing


def edges_to_cell(C2E):
    def variant(inp):
        acc = shift(inp, C2E[:])
        return reduce(acc)

        return (
            deref(shift(inp, C2E(0)))
            + deref(shift(inp, C2E(1)))
            + deref(shift(inp, C2E(2)))
            + deref(shift(inp, C2E(3)))
        )

    return variant


# Cells
# 0 1 2
# 3 4 5
# 6 7 8

# Edges
#    0   1   2
#  3   4   5   6
#    7   8   9
# 10  11  12  13
#   14  15  16
# 17  18  19  20
#   21  22  23
class Cell:
    ...


c2e_tbl = [
    [0, 4, 7, 3],
    [1, 5, 8, 4],
    [2, 6, 9, 5],
    [7, 11, 14, 10],
    [8, 12, 15, 11],
    [9, 13, 16, 12],
    [14, 18, 21, 17],
    [15, 19, 22, 18],
    [16, 20, 23, 19],
]


class C2E_indirect:
    def __init__(self, neigh_index) -> None:
        self.neigh_index = neigh_index

    def __call__(self, pos: Dict) -> Dict:
        if Cell in pos.keys():
            new_pos = pos.copy()
            del new_pos[Cell]
            new_pos[Edge] = c2e_tbl[pos[Cell]][self.neigh_index]
            return new_pos
        return pos


def test_indirect():
    inp = np_as_located_field(Edge)(np.asarray(list(range(24))))
    out = np_as_located_field(Cell)(np.zeros([9]))

    ref = np.asarray(list(sum(row) for row in c2e_tbl))

    apply_stencil(edges_to_cell(C2E_indirect), {Cell: range(9)}, [inp], [out])
    assert np.allclose(ref, np.asarray(out))


class ColorE:
    ...


class IE:
    ...


class JE:
    ...


class IC:
    ...


class JC:
    ...


class C2E_strided:
    def __init__(self, neigh_index) -> None:
        self.neigh_index = neigh_index

    def __call__(self, pos: Dict) -> Dict:
        if IC in pos.keys() and JC in pos.keys():
            new_pos = pos.copy()
            del new_pos[IC]
            del new_pos[JC]

            if self.neigh_index == 0:
                new_pos[IE] = pos[IC]
                new_pos[JE] = pos[JC]
                new_pos[ColorE] = 0
            elif self.neigh_index == 1:
                new_pos[IE] = pos[IC]
                new_pos[JE] = pos[JC] + 1
                new_pos[ColorE] = 1
            elif self.neigh_index == 2:
                new_pos[IE] = pos[IC] + 1
                new_pos[JE] = pos[JC]
                new_pos[ColorE] = 0
            elif self.neigh_index == 3:
                new_pos[IE] = pos[IC]
                new_pos[JE] = pos[JC]
                new_pos[ColorE] = 1
            else:
                raise IndexError("offset not defined")

            return new_pos
        return pos


def test_direct():
    inp = np_as_located_field(IE, JE, ColorE)(
        np.asarray(
            [
                [[0, 3], [1, 4], [2, 5], [-1, 6]],
                [[7, 10], [8, 11], [9, 12], [-1, 13]],
                [[14, 17], [15, 18], [16, 19], [-1, 20]],
                [[21, -1], [22, -1], [23, -1], [-1, -1]],
            ]
        )
    )
    out = np_as_located_field(IC, JC)(np.zeros([3, 3]))

    ref = np.asarray(list(sum(row) for row in c2e_tbl)).reshape(3, 3)

    apply_stencil(
        edges_to_cell(C2E_strided), {IC: range(3), JC: range(3)}, [inp], [out]
    )
    assert np.allclose(ref, np.asarray(out))
