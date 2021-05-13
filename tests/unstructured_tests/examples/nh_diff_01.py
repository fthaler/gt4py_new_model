from typing import Tuple


class Field:
    ...


Vertex = 1
Edge = 2
E2V = 3
Vec = 4

# Field[Vertex, Edge] ~ Field[Tuple[Vertex,Edge]] located on (Vertex x Edge)
# Field[E2V, Edge] ~ Field[Tuple[Vertex, Edge]] located on (Edge, adjacent Vertex of Edge)
# Field[E2V]
# Field[Vec] = (2, 3)
# a = Field[Vec, Vertex]
# a_vec: Field[Vec] = a(Vertex(54))
# vertex_field: Field[Vertex] = a(Vec(0))


def e2v(vertex_field: Field[Vertex]) -> Field[E2V, Edge]:
    ...


# in the following k is removed
def nh_diff_01(
    diff_multfac_smag,
    tangent_orientation: Field[Edge],
    inv_primal_edge_length: Field[Edge],
    inv_vert_vert_length: Field[Edge],
    u_vert: Field[Vertex],
    v_vert: Field[Vertex],
    primal_normal_x: Field[E2V, Edge],
    primal_normal_y: Field[E2V, Edge],
    dual_normal_x: Field[E2V, Edge],
    dual_normal_y: Field[E2V, Edge],
    vn: Field[Edge],
    smag_limit,
    smag_offset,
):
    u_vert_neighs: Field[E2V, Edge] = e2v(u_vert)
    v_vert_neighs: Field[E2V, Edge] = e2v(v_vert)

    weights_tang: Field[E2V] = local_field(e2v)(-1, 1, 0, 0)
    weights_norm: Field[E2V] = local_field(e2v)(0, 0, -1, 1)

    weights_close: Field[E2V] = local_field(e2v)(1, 1, 0, 0)
    weights_far: Field[E2V] = local_field(e2v)(0, 0, 1, 1)

    vn_vert = u_vert_neighs * primal_normal_x + v_vert_neighs * primal_normal_y
    dvt_tang = sum_reduce(E2V)(
        weights_tang
        * ((u_vert_neighs * dual_normal_x) + (v_vert_neighs * dual_normal_y))
        * tangent_orientation
    )

    dvt_norm = sum_reduce(E2V)(
        weights_norm * (u_vert_neighs * dual_normal_x + v_vert_neighs * dual_normal_y)
    )

    kh_smag_1 = pow(
        (
            sum_reduce(E2V)(weights_tang * vn_vert)
            * tangent_orientation
            * inv_primal_edge_length
        )
        + (dvt_norm * inv_vert_vert_length),
        2,
    )
    kh_smag_2 = pow(
        (
            sum_reduce(E2V)(weights_norm * vn_vert)
            * tangent_orientation
            * inv_vert_vert_length
        )
        - (dvt_tang * inv_primal_edge_length),
        2,
    )

    kh_smag = diff_multfac_smag * sqrt(kh_smag_2 + kh_smag_1)

    nabla2 = 4.0 * (
        (sum_reduce(E2V)(weights_close * vn_vert) - 2.0 * vn)
        * pow(inv_primal_edge_length, 2)
        + (sum_reduce(E2V)(weights_far * vn_vert) - 2.0 * vn)
        * pow(inv_vert_vert_length, 2)
    )
    return nabla2, kh_smag, max(0, kh_smag - smag_offset), min(kh_smag, smag_limit)


#######################################


def dvt_tang(
    weights_tang: Field[E2V],
    vert: Field[E2V, Edge, Vec],
    dual_normal: Field[E2V, Edge, Vec],
    tangent_orientation: Field[Edge],
) -> Field[Edge]:
    return (
        sum_reduce(E2V)(broadcast(Edge)(weights_tang) * dot(Vec)(vert, dual_normal))
        * tangent_orientation
    )


def dvt_norm(
    weights_norm: Field[E2V],
    vert: Field[E2V, Edge, Vec],
    dual_normal: Field[E2V, Edge, Vec],
) -> Field[Edge]:
    return sum_reduce(E2V)(weights_norm * dot(Vec)(vert, dual_normal))


def kh_smag_2(
    weights_tang: Field[E2V],
    dual_normal: Field[E2V, Edge, Vec],
    weights_norm: Field[E2V],
    vert: Field[E2V, Edge, Vec],
    vn_vert: Field[E2V, Edge],
    tangent_orientation: Field[Edge],
    inv_vert_vert_length: Field[Edge],
    inv_primal_edge_length: Field[Edge],
) -> Field[Edge]:
    dvt_tang_bound: Field[Edge] = dvt_tang(
        weights_tang,
        vert,
        dual_normal,
        tangent_orientation,
    )
    return pow(
        (
            sum_reduce(E2V)(weights_norm * vn_vert)
            * tangent_orientation
            * inv_vert_vert_length
        )
        - (dvt_tang_bound * inv_primal_edge_length),
        2,
    )


def kh_smag_1(
    weights_norm: Field[E2V],
    vert: Field[E2V, Edge, Vec],
    dual_normal: Field[E2V, Edge, Vec],
    weights_tang: Field[E2V],
    vn_vert: Field[E2V, Edge],
    tangent_orientation: Field[Edge],
    inv_primal_edge_length: Field[Edge],
    inv_vert_vert_length: Field[Edge],
) -> Field[Edge]:
    dvt_norm_bound: Field[Edge] = dvt_norm(weights_norm, vert, dual_normal)
    return pow(
        (
            sum_reduce(E2V)(weights_tang * vn_vert)
            * tangent_orientation
            * inv_primal_edge_length
        )
        + (dvt_norm_bound * inv_vert_vert_length),
        2,
    )


def kh_smag(
    diff_multfac_smag,
    vert: Field[E2V, Edge],
    dual_normal: Field[E2V, Edge, Vec],
    vn_vert: Field[E2V, Edge],
    tangent_orientation: Field[Edge],
    inv_vert_vert_length: Field[Edge],
    inv_primal_edge_length: Field[Edge],
) -> Field[Edge]:
    weights_tang: Field[E2V] = local_field(E2V)(-1, 1, 0, 0)
    weights_norm: Field[E2V] = local_field(E2V)(0, 0, -1, 1)
    return diff_multfac_smag * sqrt(
        kh_smag_2(
            weights_tang,
            vert,
            dual_normal,
            weights_norm,
            vn_vert,
            tangent_orientation,
            inv_vert_vert_length,
            inv_primal_edge_length,
        )
        + kh_smag_1(
            weights_norm,
            vert,
            dual_normal,
            weights_tang,
            vn_vert,
            tangent_orientation,
            inv_primal_edge_length,
            inv_vert_vert_length,
        )
    )


def nabla2(
    vn_vert: Field[E2V, Edge],
    vn: Field[Edge],
    inv_primal_edge_length: Field[Edge],
    inv_vert_vert_length: Field[Edge],
) -> Field[Edge]:
    weights_close: Field[E2V] = local_field(E2V)(1, 1, 0, 0)
    weights_far: Field[E2V] = local_field(E2V)(0, 0, 1, 1)
    4.0 * (
        (sum_reduce(E2V)(weights_close * vn_vert) - 2.0 * vn)
        * pow(inv_primal_edge_length, 2)
        + (sum_reduce(E2V)(weights_far * vn_vert) - 2.0 * vn)
        * pow(inv_vert_vert_length, 2)
    )


def vn_vert(
    vert: Field[E2V, Edge, Vec], primal_normal: Field[E2V, Edge, Vec]
) -> Field[E2V, Edge]:
    return dot(Vec)(vert, primal_normal)


def nh_diff_01_v2(
    diff_multfac_smag,
    tangent_orientation: Field[Edge],
    inv_primal_edge_length: Field[Edge],
    inv_vert_vert_length: Field[Edge],
    vert: Field[Vertex],
    primal_normal: Field[E2V, Edge, Vec],
    dual_normal: Field[E2V, Edge, Vec],
    vn: Field[Edge],
    smag_limit,
    smag_offset,
) -> Tuple[Field[Edge], Field[Edge], Field[Edge], Field[Edge]]:
    vert_neighs: Field[E2V, Edge] = e2v(vert)

    vn_vert_bound: Field[E2V, Edge] = vn_vert(vert_neighs, primal_normal)

    kh_smag_bound: Field[Edge] = kh_smag(
        diff_multfac_smag,
        vert_neighs,
        dual_normal,
        vn_vert_bound,
        tangent_orientation,
        inv_vert_vert_length,
        inv_primal_edge_length,
    )

    nabla2_bound: Field[Edge] = nabla2(
        vn_vert_bound,
        vn,
        inv_primal_edge_length,
        inv_vert_vert_length,
    )
    return (
        nabla2_bound,
        kh_smag_bound,
        max(0, kh_smag_bound - smag_offset),  # broadcast(Edge)(smag_offset)
        min(kh_smag_bound, smag_limit),
    )
