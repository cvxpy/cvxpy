from cvxpy.atoms.affine import hstack
from cvxpy.atoms.log_sum_exp import log_sum_exp


def quad_form_canon(expr, args):
    x = args[0]
    P = args[1]
    elems = []
    for i in range(P.shape[0]):
        for j in range(P.shape[0]):
            elems.append(P[i, j] + x[i] + x[j])
    return log_sum_exp(hstack(elems)), []
