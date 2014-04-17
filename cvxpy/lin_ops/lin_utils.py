"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

from cvxpy.lin_ops.lin_expr import LinExpr
import cvxpy.lin_ops.lin_op as lo
from cvxpy.lin_ops.lin_constraints import LinEqConstr

# Utility functions for dealing with LinExpr and LinOp.

class Counter(object):
    """A counter for ids.

    Attributes
    ----------
    count: int
        The current count.
    """
    def __init__(self):
        self.count = 0

ID_COUNTER = Counter()

def get_id():
    """Returns a new id and updates the id counter.

    Returns
    -------
    int
        A new id.
    """
    new_id = ID_COUNTER.count
    ID_COUNTER.count += 1
    return new_id

def create_var(size, scalar_coeff=1.0):
    """Creates a new internal variable.

    Parameters
    ----------
    size: tuple
        The (rows, cols) dimensions of the variable.

    Returns
    -------
    LinOP
        A LinOp representing the new variable.
    """
    return lo.LinOp(lo.EYE_MUL, get_id(), size, scalar_coeff, None)

def create_eq(lh_expr, rh_expr):
    """Creates an internal equality constraint.

    Parameters
    ----------
    lh_term: LinExpr
        The left-hand expression in the equality constraint.
    rh_term: LinExpr
        The right-hand expression in the equality constraint.

    Returns
    -------
    LinEqConstr
    """
    pass

def add_terms(lh_term, rh_term):
    """Adds two terms together.

    Parameters
    ----------
    lh_term: LinOp
        The left-hand term of the sum.
    rh_term: LinOp
        The right-hand term of the sum.

    Returns
    -------
    tuple
        (LinOp, [constraints])
    """
    # Combine identical operations.
    if lh_term.type == rh_term.type and lh_term.type in lo.IDENTICAL:
        term_sum = lo.LinOp(lh_term.type,
                            lh_term.var_id,
                            lh_term.var_size,
                            lh_term.scalar_coeff + rh_term.scalar_coeff,
                            lh_term.data)
        return (term_sum, [])
    # Sum identical types when possible.
    elif lh_term.type == rh_term.type and lh_term.type in lo.SUMMABLE:
        term_sum = lo.LinOp(lh_term.type,
                            lh_term.var_id,
                            lh_term.var_size,
                            1.0,
                            lh_term.data*lh_term.scalar_coeff + \
                            rh_term.data*rh_term.scalar_coeff)
        return (term_sum, [])
    # Split up different types by creating a new variable
    # and equality constraint.
    else:
        new_var = create_var(lh_term.size, scalar_coeff=-1.0)
        expr = LinExpr({new_var.var_id: new_var,
                        rh_term.var_id: rh_term},
                        lh_term.size)
        constr = LinEqConstr(expr, lh_term.size)
        return (lh_term, [constr])

def sum_expr(*args):
    """Sum linear expressions.

    Parameters
    ----------
    args: list
        A list of LinExpr objects.

    Returns
    -------
    LinExpr
        A LinExpr representing the sum of the args.
    """
    size = args[0].size
    new_terms = args[0].terms.copy()
    constraints = []
    for arg in args[1:]:
        for term_id, term in arg.terms.items():
            if term_id in new_terms:
                term_sum, constr = add_terms(new_terms[term_id], term)
                constraints += constr
                # TODO remove if scalar == 0.
                new_terms[term_id] = term_sum
            else:
                new_terms[term_id] = term

    return (LinExpr(terms=new_terms, size=size), constraints)
