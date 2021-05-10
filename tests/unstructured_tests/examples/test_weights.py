def sten(e2v, in_field):
    weights = local_field(e2v)(-1, 1)
    return reduce(__add__)(e2v(in_field) * weights)
