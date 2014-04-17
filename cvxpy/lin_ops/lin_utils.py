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
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
import cvxpy.interface as intf

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

def create_var(size):
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
    return lo.LinOp(lo.EYE_MUL, get_id(), size, 1.0, None)

def create_const(value, size):
    """Wraps a constant.

    Parameters
    ----------
    size: tuple
        The (rows, cols) dimensions of the variable.

    Returns
    -------
    LinOP
        A LinOp representing the new variable.
    """
    # Check if scalar.
    if size == (1, 1):
        op_type = lo.SCALAR_CONST
    # Check if sparse.
    elif intf.is_sparse(value):
        op_type = lo.SPARSE_CONST
    else:
        op_type = lo.DENSE_CONST
    return lo.LinOp(op_type, lo.CONSTANT_ID, size, 1.0, value)

def sum_expr(expressions):
    """Add linear expressions.

    Parameters
    ----------
    expression: list
        A list of linear expressions.

    Returns
    -------
    LinExpr
        A LinExpr representing the sum of the expressions.
    """
    terms = []
    for expr in expressions:
        terms += expr.terms
    return LinExpr(terms, expressions[0].size)

def neg_term(term):
    """Negate a term.

    Parameters
    ----------
    term: LinOp
        The term to be negated.

    Returns
    -------
    LinOp
        The negated term.
    """
    return lo.LinOp(term.type,
                    term.var_id,
                    term.var_size,
                    -term.scalar_coeff,
                    term.data)

def neg_expr(expr):
    """Negate an expression.

    Parameters
    ----------
    expr: LinExpr
        The expression to be negated.

    Returns
    -------
    LinExpr
        The negated expression.
    """
    new_terms = [neg_term(term) for term in expr.terms]
    return LinExpr(new_terms, expr.size)

def mul_term(constant, term):
    """Multiply a term on the left by a constant.

    Parameters
    ----------
    constant: LinOp
        The constant to multiply by.
    term: LinOp
        The term to be multiplied.

    Returns
    -------
    LinOp
        The product.
    """
    type_, var_id, var_size, scalar_coeff, data = term
    if constant.type is lo.SCALAR_CONST:
        scalar_coeff *= constant.data*constant.scalar_coeff
    else:
        type_ = lo.MUL_TYPE[(constant.type, term.type)]
        if term.type is lo.EYE_MUL:
            data = constant.data
        else:
            data = constant.data*term.data

    return lo.LinOp(type_,
                    var_id,
                    var_size,
                    scalar_coeff,
                    data)

def mul_valid(constant, expr):
    """Can the terms in the expression be multiplied by the constant?
    """
    if constant.type is lo.SCALAR_CONST:
        return True
    else:
        for term in expr.terms:
            if (constant.type, term.type) not in lo.MUL_TYPE:
                return False
    return True

def mul_expr(constant, expr, size):
    """Multiply an expression on the left by a constant.

    Parameters
    ----------
    constant: LinOp
        A constant or parameter.
    expr: LinExpr
        A linear expression.
    size: tuple
        The size of the product.

    Returns
    -------
    tuple
        (LinExpr for product, list of constraints)
    """
    constraints = []
    if mul_valid(constant, expr):
        new_terms = [mul_term(constant, t) for t in expr.terms]
    # Cannot multiply. Make a new variable and constraint.
    else:
        new_var = create_var(expr.size)
        new_terms = [mul_term(constant, new_var)]
        constr_terms = expr.terms[:] + [neg_term(new_var)]
        constr_expr = LinExpr(constr_terms, expr.size)
        constraints.append(LinEqConstr(constr_expr, expr.size))
    return (LinExpr(new_terms, size), constraints)

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
    expr = sum_expr([lh_expr, neg_expr(rh_expr)])
    return LinEqConstr(expr, lh_expr.size)

def create_leq(lh_expr, rh_expr):
    """Creates an internal less than or equal constraint.

    Parameters
    ----------
    lh_term: LinExpr
        The left-hand expression in the <= constraint.
    rh_term: LinExpr
        The right-hand expression in the <= constraint.

    Returns
    -------
    LinEqConstr
    """
    expr = sum_expr([lh_expr, neg_expr(rh_expr)])
    return LinLeqConstr(expr, lh_expr.size)

def get_expr_vars(expr):
    """Get a list of the unique variables in the expression and their sizes.

    Parameters
    ----------
    expr: LinExpr
        The expression to extract the variables from.

    Returns
    -------
    list
        A list of (var id, var size) pairs, where each var id is unique.
    """
    vars_ = set()
    for term in expr.terms:
        if term.var_id is not lo.CONSTANT_ID:
            vars_.add((term.var_id, term.var_size))
    return list(vars_)

# def add_terms(lh_term, rh_term):
#     """Adds two terms together.

#     Parameters
#     ----------
#     lh_term: LinOp
#         The left-hand term of the sum.
#     rh_term: LinOp
#         The right-hand term of the sum.

#     Returns
#     -------
#     LinOp
#         A LinOp representing the sum of the terms or None if can't be added.
#     """
#     # Combine identical operations.
#     if lh_term.type == rh_term.type and lh_term.type in lo.IDENTICAL:
#         term_sum = lo.LinOp(lh_term.type,
#                             lh_term.var_id,
#                             lh_term.var_size,
#                             lh_term.scalar_coeff + rh_term.scalar_coeff,
#                             lh_term.data)
#     # Sum identical types when possible.
#     elif lh_term.type == rh_term.type and lh_term.type in lo.SUMMABLE:
#         term_sum = lo.LinOp(lh_term.type,
#                             lh_term.var_id,
#                             lh_term.var_size,
#                             1.0,
#                             lh_term.data*lh_term.scalar_coeff + \
#                             rh_term.data*rh_term.scalar_coeff)
#     # Split up different types by creating a new variable
#     # and equality constraint.
#     # TODO dense + sparse/eye
#     else:
#         term_sum = None
#     return term_sum
