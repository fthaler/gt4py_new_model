def sten(c2c, field_in):
    return reduce(__add__)(broadcast(c2c)(field_in) + c2c(field_in))
