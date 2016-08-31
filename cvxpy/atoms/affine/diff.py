from cvxpy.expressions.expression import Expression


def diff(x, k=1, axis=0):
    """ Vector of kth order differences.

    Takes in a vector of length n and returns a vector
    of length n-k of the kth order differences.

    diff(x) returns the vector of differences between
    adjacent elements in the vector, that is

    [x[2] - x[1], x[3] - x[2], ...]

    diff(x, 2) is the second-order differences vector,
    equivalently diff(diff(x))

    diff(x, 0) returns the vector x unchanged
    """
    x = Expression.cast_to_const(x)
    if axis == 1:
        x = x.T
    m, n = x.size
    if k < 0 or k >= m:
        raise ValueError('Must have k >= 0 and X must have < k elements along axis')

    d = x
    for i in range(k):
        d = d[1:, :] - d[:-1, :]

    if axis == 1:
        return d.T
    else:
        return d
