from unstructured.concepts import neighborhood, LocationType
from unstructured.helpers import simple_connectivity


@neighborhood(LocationType.Vertex, LocationType.Vertex)
class CartesianNeighborHood:
    pass


@simple_connectivity(CartesianNeighborHood())
def cartesian_connectivity(*indices):
    class neighs:
        def __getitem__(self, neighindices):
            if not isinstance(neighindices, tuple):
                neighindices = (neighindices,)
            return tuple(
                map(lambda x: x[0] + x[1], zip(indices, neighindices)),
            )

    return neighs()
