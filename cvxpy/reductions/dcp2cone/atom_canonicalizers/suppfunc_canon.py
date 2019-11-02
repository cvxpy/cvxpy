from cvxpy import hstack, SOC, sum
from cvxpy.constraints.exponential import ExpCone
import numpy as np
from scipy.sparse import csc_matrix


def scspsdvec2mat(vec):
    n = int(np.sqrt(vec.size * 2))
    rows, cols = np.triu_indices(n)
    mats = []
    for i in range(vec.size):
        r, c = rows[i], cols[i]
        mat = np.zeros(shape=(n, n))
        if r == c:
            mat[r, r] = 1
        else:
            mat[r, c] = 1 / np.sqrt(2)
            mat[c, r] = 1 / np.sqrt(2)
        mat = vec[i] * mat
        mats.append(mat)
    V = sum(mats)
    return V


def selector_matrix(selector, inshape):
    """
    :param selector: a vector of nonnegative integer indices.
    :param inshape: an integer > np.max(selector)

    :return: A csc-format sparse matrix ``mat``, where for some cvxpy expression ``x``,
    ``expr1 = x[selector]`` and ``expr2 = mat @ x`` are equivalent in a symbolic sense.

    This function is necessary, because special-indexing like "x[selector]" doesn't
    have a graph_implementation function, and the usual CVXPY rewriting system to handle
    such indexing can't be relied at this point in the compilation process.
    """
    vals = np.ones(selector.size)
    rows = np.arange(selector.size)
    mat = csc_matrix((vals, (rows, selector)), shape=(selector.size, inshape))
    return mat


def suppfunc_canon(expr, args):
    y = args[0].flatten()
    # ^ That's the user-supplied argument to the support function.
    A, b = args[1], args[2]
    K_sels = b.K_sels
    # ^ That defines the set "X" associated with this support function.
    eta = args[3]
    # ^ Variable, of shape (b.size,). It's the main part of the duality
    # trick for representing the epigraph of this support function.
    n = A.shape[1]
    n0 = y.size
    if n > n0:
        # The description of the set "X" used in this support
        # function included n - n0 > 0 auxiliary variables.
        # We can pretend these variables were user-defined
        # by appending a suitable number of zeros to y.
        y_lift = hstack([y, np.zeros(shape=(n - n0,))])
    else:
        y_lift = y
    local_cons = [A.T @ eta + y_lift == 0]
    # now, the main conic constraints on eta.
    #   nonneg, exp, soc, psd
    nonnegsel = K_sels['nonneg']
    if nonnegsel.size > 0:
        selector = selector_matrix(nonnegsel, eta.size)
        temp_expr = selector @ eta
        local_cons.append(temp_expr >= 0)
    socsels = K_sels['soc']
    for socsel in socsels:
        selector = selector_matrix(np.array([socsel[0]]), eta.size)
        tempsca = selector @ eta
        selector = selector_matrix(socsel[1:], eta.size)
        tempvec = selector @ eta
        soccon = SOC(tempsca, tempvec)
        local_cons.append(soccon)
    psdsels = K_sels['psd']
    for psdsel in psdsels:
        selector = selector_matrix(psdsel, eta.size)
        curvec = selector @ eta
        curmat = scspsdvec2mat(curvec)
        local_cons.append(curmat >> 0)
    expsel = K_sels['exp']
    if expsel.size > 0:
        matexpsel = np.reshape(expsel, (-1, 3))
        selector = selector_matrix(matexpsel[:, 0], eta.size)
        curr_u = selector @ eta
        selector = selector_matrix(matexpsel[:, 1], eta.size)
        curr_v = selector @ eta
        selector = selector_matrix(matexpsel[:, 2], eta.size)
        curr_w = selector @ eta
        # (curr_u, curr_v, curr_w) needs to belong to the dual
        # exponential cone, as used by the SCS solver. We map
        # this to a primal exponential cone as follows.
        ec = ExpCone(-curr_v, -curr_u, np.exp(1) * curr_w)
        local_cons.append(ec)
    epigraph = b @ eta
    return epigraph, local_cons
