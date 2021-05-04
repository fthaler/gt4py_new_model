from unstructured.concepts import LocationType, apply_stencil, field_dec, if_
from unstructured.helpers import array_to_field, as_1d, as_2d
import math
import numpy as np
from .hdiff_reference import hdiff_reference


def make_fivepoint(shape_2d):
    strides = [1, shape_2d[1]]

    class v2v_conn:
        center = 0
        right = 1
        left = 2
        bottom = 3
        top = 4

        def __call__(self, field):
            class acc:
                def __getitem__(self, neigh_index):
                    @field_dec(LocationType.Vertex)
                    def fp_field(field_index):
                        if neigh_index == 0:
                            return field(field_index)
                        elif neigh_index == 1:
                            return field(field_index + strides[0])
                        elif neigh_index == 2:
                            return field(field_index - strides[0])
                        elif neigh_index == 3:
                            return field(field_index + strides[1])
                        elif neigh_index == 4:
                            return field(field_index - strides[1])
                        else:
                            assert False

                    return fp_field

            return acc()

    return v2v_conn()


def laplacian(conn, inp):
    five_point = conn(inp)
    return -4 * five_point[conn.center] + (
        five_point[conn.right]
        + five_point[conn.left]
        + five_point[conn.top]
        + five_point[conn.bottom]
    )


def hdiff_flux_x(conn, inp):
    lap = conn(laplacian(conn, inp))
    flux = lap[conn.center] - lap[conn.right]

    neighs = conn(inp)
    return if_(flux * (neighs[conn.right] - neighs[conn.center]) > 0, lambda x: 0, flux)


def hdiff_flux_y(conn, inp):
    lap = conn(laplacian(conn, inp))
    flux = lap[conn.center] - lap[conn.bottom]

    neighs = conn(inp)
    return if_(
        flux * (neighs[conn.bottom] - neighs[conn.center]) > 0, lambda x: 0, flux
    )


def hdiff(conn, inp, coeff):
    flx = conn(hdiff_flux_x(conn, inp))
    fly = conn(hdiff_flux_y(conn, inp))
    return inp - (
        coeff * (flx[conn.center] - flx[conn.left] + fly[conn.center] - fly[conn.top])
    )


def test_hdiff(hdiff_reference):
    inp, coeff, out = hdiff_reference
    shape = (inp.shape[0], inp.shape[1])
    inp_s = array_to_field(as_1d(inp[:, :, 0]), LocationType.Vertex)
    coeff_full_domain = np.zeros(shape)
    coeff_full_domain[2:-2, 2:-2] = coeff[:, :, 0]
    coeff_s = array_to_field(as_1d(coeff_full_domain), LocationType.Vertex)
    out_s = as_1d(np.zeros_like(inp)[:, :, 0])

    domain = np.arange(math.prod(shape))
    domain_2d = as_2d(domain, shape)
    inner_domain = [as_1d(domain_2d[2:-2, 2:-2]).tolist()]

    apply_stencil(
        hdiff,
        inner_domain,
        [make_fivepoint(shape), inp_s, coeff_s],
        [out_s],
    )

    assert np.allclose(out[:, :, 0], np.asarray(as_2d(out_s, shape)[2:-2, 2:-2]))
