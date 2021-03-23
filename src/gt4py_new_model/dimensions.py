from dataclasses import dataclass


@dataclass(frozen=True)
class Dimension:
    dimension: str
    offset: int = 0

    def __add__(self, other):
        if isinstance(other, Dimension):
            assert other.dimension == self.dimension
            return Dimension(
                dimension=self.dimension, offset=self.offset + other.offset
            )
        return Dimension(dimension=self.dimension, offset=self.offset + other)

    def __sub__(self, other):
        if isinstance(other, Dimension):
            assert other.dimension == self.dimension
            return Dimension(
                dimension=self.dimension, offset=self.offset - other.offset
            )
        return Dimension(dimension=self.dimension, offset=self.offset - other)

    @staticmethod
    def collect(*indices: "Dimension") -> tuple["Dimension"]:
        res = dict[str, int]()
        for i in indices:
            res[i.dimension] = i.offset + res.get(i.dimension, 0)
        return tuple(Dimension(dimension=k, offset=v) for k, v in res.items())


I = Dimension("i")
J = Dimension("j")
K = Dimension("k")
