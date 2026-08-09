import numpy as np

from cvxpy import SOC, Variable, hstack, multiply, reshape
from cvxpy.atoms.affine.transpose import swapaxes
from cvxpy.atoms.affine.upper_tri import upper_tri
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.psd import PSD, SvecPSD
from cvxpy.expressions.expression import Expression
from cvxpy.utilities.psd_utils import TriangleKind
from cvxpy.utilities.solver_context import SolverInfo


def _svec_psd_dual_arg(
    arg: Expression, n: int, num_cones: int, solver_context: SolverInfo,
) -> Expression:
    """Map solver-formatted svec rows to their Euclidean dual cone.

    Svec is self-dual with sqrt(2) off-diagonal scaling. Without that
    scaling, the dual cone needs a factor of 1/2 on off-diagonal entries.
    """
    if solver_context.psd_sqrt2_scaling:
        return arg

    # NumPy enumerates triangle indices row-by-row, while solver svec entries
    # are ordered column-by-column. Column-major lower-triangle order has the
    # same diagonal pattern as row-major upper-triangle order, and vice versa,
    # so use the opposite triangle when constructing these weights.
    if solver_context.psd_triangle_kind == TriangleKind.LOWER:
        rows, cols = np.triu_indices(n)
    else:
        rows, cols = np.tril_indices(n)
    weights = np.where(rows == cols, 1.0, 0.5)
    weights = np.tile(weights, num_cones)
    return multiply(weights, arg)


def suppfunc_canon(expr, args, solver_context: SolverInfo | None = None):
    y = args[0].flatten(order="F")
    # ^ That's the user-supplied argument to the support function.
    parent = expr._parent
    if solver_context is None:
        raise ValueError("SuppFunc canonicalization requires solver context.")
    A, b, K_sels = parent._conic_repr_of_set(solver_context)
    # ^ That defines the set "X" associated with this support function.
    eta = Variable(shape=(b.size,))
    expr._eta = eta
    # ^ The main part of the duality trick for representing the epigraph
    # of this support function.
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
    # now, the conic constraints on eta.
    #   nonneg, exp, soc, psd
    nonnegsel = K_sels["nonneg"]
    if nonnegsel.size > 0:
        temp_expr = eta[nonnegsel]
        local_cons.append(temp_expr >= 0)
    socsels = K_sels["soc"]
    for socsel in socsels:
        tempsca = eta[socsel[0]]
        tempvec = eta[socsel[1:]]
        soccon = SOC(tempsca, tempvec)
        local_cons.append(soccon)
    psdsels = K_sels["psd"]
    for psdsel, source_con in psdsels:
        eta_block = eta[psdsel]
        if isinstance(source_con, SvecPSD):
            n = source_con.cone_sizes()[0]
            dual_arg = _svec_psd_dual_arg(
                eta_block, n, source_con.num_cones(), solver_context)
            local_cons.append(SvecPSD(dual_arg, n=n))
        else:
            eta_mat = reshape(eta_block, source_con.shape, order='F')
            local_cons.append(
                upper_tri(eta_mat) == upper_tri(swapaxes(eta_mat, -2, -1)))
            local_cons.append(PSD(eta_mat))
    expsel = K_sels["exp"]
    if expsel.size > 0:
        matexpsel = np.reshape(expsel, (-1, 3))
        curr_u = eta[matexpsel[:, 0]]
        curr_v = eta[matexpsel[:, 1]]
        curr_w = eta[matexpsel[:, 2]]
        # (curr_u, curr_v, curr_w) belongs to the dual exponential cone.
        # Map it to CVXPY's primal exponential cone convention.
        ec = ExpCone(-curr_v, -curr_u, np.exp(1) * curr_w)
        local_cons.append(ec)
    epigraph = b @ eta
    return epigraph, local_cons
