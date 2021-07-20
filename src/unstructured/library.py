from unstructured.builtins import reduce


def sum(fun=None):
    if fun is None:
        return reduce(lambda a, b: a + b, 0)
    else:
        return reduce(
            lambda first, a, b: first + fun(a, b), 0
        )  # TODO tracing for *args
