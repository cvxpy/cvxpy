from cvxpy.atoms.elementwise.abs import abs
from cvxpy.atoms.elementwise.power import power
from cvxpy.expressions.variables.variable import Variable
from cvxpy.reductions.dcp2cone.atom_canonicalizers.abs_canon import abs_canon
from cvxpy.reductions.dcp2cone.atom_canonicalizers.power_canon import power_canon


def huber_canon(expr, args):
    M = expr.M
    x = args[0]
    shape = expr.shape
    n = Variable(*shape)
    s = Variable(*shape)

    # n**2 + 2*M*|s|
    # TODO(akshayka): Make use of recursion inherent to canonicalization
    # process and just return a power / abs expressions for readability sake
    power_expr = power(n, 2)
    n2, constr_sq = power_canon(power_expr, power_expr.args)
    abs_expr = abs(s)
    abs_s, constr_abs = abs_canon(abs_expr, abs_expr.args)
    obj = n2 + 2 * M * abs_s

    # x == s + n
    constraints = constr_sq + constr_abs
    constraints.append(x == s + n)
    return obj, constraints
