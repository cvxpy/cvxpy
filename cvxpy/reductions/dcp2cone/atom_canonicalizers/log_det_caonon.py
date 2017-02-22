from cvxpy.atoms.affine.diag import diag_mat, diag_vec
from cvxpy.atoms.affine.trace import trace
from cvxpy.atoms.affine.transpose import transpose
from cvxpy.atoms.affine.upper_tri import upper_tri
from cvxpy.atoms.elemwise.log import log
from cvxpy.constraints.semidefinite import SDP
from cvxpy.expressions.variables.variable import Variable
from cvxpy.expressions.variables.semidef_var import Semidef
from cvxpy.reductions.dcp2cone.atom_canonicalizers.log_canon import log_canon


def log_det_canon(expr, args):
    """Reduces the atom to an affine expression and list of constraints.

    Creates the equivalent problem::

       maximize    sum(log(D[i, i]))
       subject to: D diagonal
                   diag(D) = diag(Z)
                   Z is upper triangular.
                   [D Z; Z.T A] is positive semidefinite

    The problem computes the LDL factorization:

    .. math::

       A = (Z^TD^{-1})D(D^{-1}Z)

    This follows from the inequality:

    .. math::

       \det(A) >= \det(D) + \det([D, Z; Z^T, A])/\det(D)
               >= \det(D)

    because (Z^TD^{-1})D(D^{-1}Z) is a feasible D, Z that achieves
    det(A) = det(D) and the objective maximizes det(D).

    Parameters
    ----------
    expr : log_det
    args : list
        The arguments for the expression

    Returns
    -------
    tuple
        (Variable for objective, list of constraints)
    """
    A = args[0]  # n by n matrix.
    n, _ = A.shape
    X = Variable(2*n, 2*n)
    # TODO(akshayka): Write an implementation for Semidef?
    X, constraints = Semidef(2*n).canonical_form
    Z = Variable(n, n)
    D = Variable(n, 1)
    # Require that X and A are PSD.
    constraints += [SDP(A)]
    # Fix Z as upper triangular, D as diagonal,
    # and diag(D) as diag(Z).
    Z_lower_tri = upper_tri(transpose(Z))

    # TODO(akshayka): What's the point of create_eq here?
    constraints.append(lu.create_eq(Z_lower_tri))
    # D[i, i] = Z[i, i]
    constraints.append(D == diag_mat(Z))
    # Fix X using the fact that A must be affine by the DCP rules.
    # X[0:n, 0:n] == D
    constraints.append(X[0:n, 0:n] == diag_vec(D))
    # X[0:n, n:2*n] == Z,
    constraints.append(X[0:n, n:2*n] == Z)
    # X[n:2*n, n:2*n] == A
    constraints.append(X[n:2*n, n:2*n] == A)
    # Add the objective sum(log(D[i, i])
    log_expr = log(D) obj, constr = log_canon(log_expr, log_expr.args)
    constraints += constr
    return sum_entries(obj), constraints
