from unstructured.concepts import LocationType


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
