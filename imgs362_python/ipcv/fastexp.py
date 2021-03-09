def bits_of(m):
    """
    Binary digits of n, from the least significant bit.

    >>> list(bits_of(11))
    [1, 1, 0, 1]
    """
    n = int(m)
    while n:
        yield n & 1
        n >>= 1


def fast_exp(x, n):
    """
    Compute x^n, using the binary expansion of n.
    """
    result = 1
    partial = x

    for bit in bits_of(n):
        if bit:
            result *= partial
        partial **= 2

    return result