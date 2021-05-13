import pytest
import operator
import numpy as np

from unstructured.concepts import sum_reduce
from unstructured.utils import axis
from unstructured.helpers import array_as_field


@axis()
class GridDim:
    pass


class TestIndex:
    def __getitem__(self, index):
        print(index.__index__())


def test_index():
    TestIndex()[GridDim(4)]


class GridIndexField:
    def __getitem__(self, index):
        if not isinstance(index, GridDim):
            raise TypeError()
        return index.__index__()


@axis(length=2)
class Vec2Dim:
    pass


class Vec2Field:
    def __init__(self, val1, val2):
        self.val = [val1, val2]

    def __getitem__(self, index):
        if not isinstance(index, Vec2Dim):
            raise TypeError()
        return self.val[index.__index__()]


def test_indexing():
    assert GridIndexField()[GridDim(5)] == 5
    with pytest.raises(TypeError):
        GridIndexField()[Vec2Dim(0)]

    a_vec2 = Vec2Field(42, 43)
    assert a_vec2[Vec2Dim(0)] == 42
    assert a_vec2[Vec2Dim(1)] == 43
    with pytest.raises(IndexError):
        operator.index(Vec2Dim(2))

    assert Vec2Dim(1) != GridDim(1)
    assert Vec2Dim(1) == Vec2Dim(1)
    assert Vec2Dim(1) != Vec2Dim(0)


test_indexing()


def idx(cls):
    def _idx(indices):
        for ind in indices:
            if isinstance(ind, cls):
                return ind.__index__()
        raise TypeError()

    return _idx


class TupleIndexGridVec2Field:
    axises = (GridDim, Vec2Dim)

    def __getitem__(self, indices):
        if isinstance(indices, tuple):
            # TODO generalize
            if isinstance(indices[0], GridDim):
                if not isinstance(indices[1], Vec2Dim):
                    raise TypeError()
            elif isinstance(indices[0], Vec2Dim):
                if not isinstance(indices[1], GridDim):
                    raise TypeError()
            else:
                raise TypeError()

            return (idx(GridDim)(indices), idx(Vec2Dim)(indices))
        else:
            if isinstance(indices, GridDim):
                return Vec2Field((indices.__index__(), 0), (indices.__index__(), 1))
            elif isinstance(indices, Vec2Dim):

                class _grid_field:
                    def __getitem__(self, index):
                        if not isinstance(index, GridDim):
                            raise TypeError()
                        return (index.__index__(), indices.__index__())

                return _grid_field()
            else:
                raise TypeError()


def test_GridVec2Field():
    field = TupleIndexGridVec2Field()
    assert field[GridDim(5), Vec2Dim(0)] == (5, 0)
    assert field[GridDim(5)][Vec2Dim(0)] == (5, 0)
    assert field[Vec2Dim(0)][GridDim(5)] == (5, 0)


test_GridVec2Field()


def tupelize(tup):
    if isinstance(tup, tuple):
        return tup
    else:
        return (tup,)


def test_np_slice():
    grid_vec2 = np.zeros((42, 2))
    my_field = array_as_field(GridDim, Vec2Dim)(grid_vec2)

    print(my_field[GridDim(23), Vec2Dim(0)])
    print(my_field[Vec2Dim(1), GridDim(23)])
    slized = my_field[Vec2Dim(1)]
    print(slized)
    slized[GridDim(41)]
    print(my_field[Vec2Dim(1)][GridDim(24)])


def test_reduce():
    grid_vec2 = np.ones((42, 2))
    my_field = array_as_field(GridDim, Vec2Dim)(grid_vec2)

    reduced_field = sum_reduce(Vec2Dim)(my_field)
    for axis in reduced_field.axises:
        print(str(axis(0)))

    print(reduced_field[GridDim(2)])


test_reduce()
