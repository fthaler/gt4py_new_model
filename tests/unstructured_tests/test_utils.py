from unstructured.utils import split_indices, axis


@axis()
class first:
    pass


@axis()
class second:
    pass


def test_split_indices():
    indices = (first(0), second(1), first(2))
    first_indices, rest = split_indices(indices, (first, second))
    assert first_indices == (first(0), second(1))
    assert rest == (first(2),)
