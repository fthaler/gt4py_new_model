import enum


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
                assert field.location == self.in_location

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


def lift(stencil):
    def lifted(acc):
        class wrap:
            def __getitem__(self, i):
                return stencil(acc[i])

        return wrap()

    return lifted


def neighborhood(in_loc, out_loc):
    """syntactic sugar to create a neighborhood"""

    def impl(cls):
        cls.in_location = in_loc
        cls.out_location = out_loc
        return cls

    return impl


def ufield(loc):
    def _impl(fun):
        class _field:
            location = loc

            def __call__(self, index):
                return fun(index)

        return _field()

    return _impl


def apply_stencil(
    stencil, domain, connectivities, out, ins
):  # TODO support multiple input/output
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
    for inp, nh_chains in zip(ins, stencil.acc_neighborhood_chains):
        if nh_chains:
            cur_conn = None
            for nh in nh_chains:
                if not cur_conn:
                    cur_conn = conn_map[neighborhood_to_key(nh)]
                else:
                    cur_conn = conn_mult(cur_conn, conn_map[neighborhood_to_key(nh)])
            stencil_args.append(cur_conn(inp))
        else:
            stencil_args.append(inp)

    for i in domain:
        out[i] = stencil(*map(lambda fun: fun(i), stencil_args))


@enum.unique
class LocationType(enum.IntEnum):
    Vertex = 0
    Edge = 1
    Cell = 2
