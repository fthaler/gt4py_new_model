from unstructured.concepts import LocationType, connectivity, ufield


def as_1d(arr):
    return arr.flatten()


def as_2d(arr, shape):
    return arr.reshape(*shape)


# a field is a function from index to element `()` not `[]`
# (or change the conn)
def as_field(arr, loc: LocationType):
    class _field:
        location = loc

        def __call__(self, i: int):
            return arr[i]

    return _field()


def simple_connectivity(neighborhood):
    def _impl(fun):  # fun is function from index to array of neighbor index
        @connectivity(neighborhood)
        def conn(field):
            @ufield(neighborhood.in_location)
            def _field(index):
                return [field(i) for i in fun(index)]

            return _field

        return conn

    return _impl
