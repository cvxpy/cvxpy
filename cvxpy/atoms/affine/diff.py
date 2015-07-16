from cvxpy.expressions.expression import Expression

def diff(x, k=1):
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
    m, n = x.size

    try:
        assert 0 <= k < m
        assert n == 1
    except:
        raise ValueError('Must have k >= 0 and x must be a 1D vector with < k elements')

    d = x
    for i in range(k):
        d = d[1:] - d[:-1]
        
    return d