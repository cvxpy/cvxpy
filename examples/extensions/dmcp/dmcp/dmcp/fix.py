__author__ = 'Xinyue'
from cvxpy import *

def fix_prob(prob, fix_var):
    """Fix the given variables in the problem.

        Parameters
        ----------
        expr : Problem
        fix_var : List
            Variables to be fixed.

        Returns
        -------
        Problem
        """
    new_cost = fix(prob.objective.args[0], fix_var)
    if prob.objective.NAME == 'minimize':
        new_obj = Minimize(new_cost)
    else:
        new_obj = Maximize(new_cost)
    new_constr = []
    for con in prob.constraints:
        left = fix(con.args[0],fix_var)
        right = fix(con.args[1],fix_var)
        if con.OP_NAME == "<=":
            new_constr.append(left <= right)
        elif con.OP_NAME == ">>":
            new_constr.append(left >> right)
        else:
            new_constr.append(left == right)
    new_prob = Problem(new_obj, new_constr)
    return new_prob


def fix(expr, fix_var):
    """Fix the given variables in the expression.

        Parameters
        ----------
        expr : Expression
        fix_var : List
            Variables to be fixed.

        Returns
        -------
        Expression
    """
    fix_var_id = [var.id for var in fix_var]
    if isinstance(expr, Variable) and expr.id in fix_var_id:
        para = Parameter(expr.size[0], expr.size[1], sign=expr.sign)
        if expr.sign == "POSITIVE":
            para.value = abs(expr).value
        elif expr.sign == "NEGATIVE":
            para.value = -abs(expr).value
        else:
            para.value = expr.value
        return para
    elif len(expr.args) == 0:
        return expr
    else:
        new_args = []
        for arg in expr.args:
            new_args.append(fix(arg, fix_var))
        return expr.copy(new_args)
