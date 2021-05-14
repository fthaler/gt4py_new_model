import math
import numpy as np
from unstructured.concepts import (
    apply_stencil,
    element_access_to_field,
    if_,
)
from unstructured.helpers import (
    array_as_field,
    as_1d,
    as_2d,
    constant_field,
)

from unstructured.utils import (
    get_index_of_type,
    print_axises,
    remove_indices_of_axises,
    axis,
)
from .hdiff_reference import hdiff_reference


@axis()
class Vertex:
    pass


@axis(length=5, aliases=["center", "right", "left", "bottom", "top"])
class FP:
    pass


def test_FP():
    assert FP.center == FP(0)


def make_fivepoint(shape_2d):
    strides = [1, shape_2d[1]]

    class v2v_conn:
        def __call__(self, field):
            def shift(field_index, fp_index):
                if fp_index == FP.center:
                    return field_index.__index__()
                elif fp_index == FP.right:
                    return field_index.__index__() + strides[0]
                elif fp_index == FP.left:
                    return field_index.__index__() - strides[0]
                elif fp_index == FP.bottom:
                    return field_index.__index__() + strides[1]
                elif fp_index == FP.top:
                    return field_index.__index__() - strides[1]
                else:
                    assert False

            @element_access_to_field(
                axises=field.axises + (FP,), element_type=field.element_type
            )
            def elem_access(indices):
                v_index = get_index_of_type(Vertex)(indices)
                fp_index = get_index_of_type(FP)(indices)

                replaced_index = Vertex(shift(v_index, fp_index))
                new_indices = (replaced_index,) + remove_indices_of_axises(
                    (FP, Vertex), indices
                )
                return field[new_indices]

            return elem_access

    return v2v_conn()


def laplacian(conn, inp):
    print_axises(inp.axises)
    five_point = conn(inp)
    return -4 * five_point[FP.center] + (
        five_point[FP.right]
        + five_point[FP.left]
        + five_point[FP.top]
        + five_point[FP.bottom]
    )


def hdiff_flux_x(conn, inp):
    lap = conn(laplacian(conn, inp))
    flux = lap[FP.center] - lap[FP.right]

    neighs = conn(inp)
    return if_(
        flux * (neighs[FP.right] - neighs[FP.center]) > constant_field(Vertex)(0.0),
        constant_field(Vertex)(0.0),
        flux,
    )


def hdiff_flux_y(conn, inp):
    lap = conn(laplacian(conn, inp))
    flux = lap[FP.center] - lap[FP.bottom]

    neighs = conn(inp)
    return if_(
        flux * (neighs[FP.bottom] - neighs[FP.center]) > constant_field(Vertex)(0.0),
        constant_field(Vertex)(0.0),
        flux,
    )


def hdiff(conn, inp, coeff):
    flx = conn(hdiff_flux_x(conn, inp))
    fly = conn(hdiff_flux_y(conn, inp))
    return inp - (
        coeff * (flx[FP.center] - flx[FP.left] + fly[FP.center] - fly[FP.top])
    )


def test_hdiff(hdiff_reference):
    inp, coeff, out = hdiff_reference
    shape = (inp.shape[0], inp.shape[1])
    inp_s = array_as_field(Vertex)(as_1d(inp[:, :, 0]))
    coeff_full_domain = np.zeros(shape)
    coeff_full_domain[2:-2, 2:-2] = coeff[:, :, 0]
    coeff_s = array_as_field(Vertex)(as_1d(coeff_full_domain))
    out_s = as_1d(np.zeros_like(inp)[:, :, 0])

    domain = np.arange(math.prod(shape))
    domain_2d = as_2d(domain, shape)
    inner_domain = [(as_1d(domain_2d[2:-2, 2:-2]).tolist(), Vertex)]

    apply_stencil(
        hdiff,
        inner_domain,
        [make_fivepoint(shape), inp_s, coeff_s],
        [out_s],
    )

    assert np.allclose(out[:, :, 0], np.asarray(as_2d(out_s, shape)[2:-2, 2:-2]))
