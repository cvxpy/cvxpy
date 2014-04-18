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
    count : int
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

def create_var(size, var_id=None):
    """Creates a new internal variable.

    Parameters
    ----------
    size : tuple
        The (rows, cols) dimensions of the variable.
    var_id : int
        The id of the variable.

    Returns
    -------
    LinOP
        A LinOp representing the new variable.
    """
    if var_id is None:
        var_id = get_id()
    return lo.LinOp(lo.EYE_MUL, var_id, size, 1.0, None)

def create_var_expr(size, var_id=None):
    """Creates an expression containing a single new internal variable.

    Parameters
    ----------
    size : tuple
        The (rows, cols) dimensions of the variable.
    var_id : int
        The id of the variable.

    Returns
    -------
    LinExpr
        A LinExpr representing the new variable.
    """
    new_var = create_var(size, var_id)
    return LinExpr([new_var], size)

def create_param(value, size):
    """Wraps a parameter.

    Parameters
    ----------
    value : CVXPY Expression
        A function of parameters.
    size : tuple
        The (rows, cols) dimensions of the expression.

    Returns
    -------
    LinOP
        A LinOp wrapping the parameter.
    """
    return lo.LinOp(lo.PARAM, lo.CONSTANT_ID, size, 1.0, value)

def create_param_expr(value, size):
    """Creates an expression with a single parameter.

    Parameters
    ----------
    value : CVXPY Expression
        A function of parameters.
    size : tuple
        The (rows, cols) dimensions of the expression.

    Returns
    -------
    LinExpr
        A LinExpr wrapping the parameter.
    """
    param = create_param(value, size)
    return LinExpr([param], size)

def create_const(value, size):
    """Wraps a constant.

    Parameters
    ----------
    value : scalar, NumPy matrix, or SciPy sparse matrix.
        The numeric constant to wrap.
    size : tuple
        The (rows, cols) dimensions of the constant.

    Returns
    -------
    LinOP
        A LinOp wrapping the constant.
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

def create_const_expr(value, size):
    """Creates an expression with a single constant.

    Parameters
    ----------
    value : scalar, NumPy matrix, or SciPy sparse matrix.
        The numeric constant to wrap.
    size : tuple
        The (rows, cols) dimensions of the constant.

    Returns
    -------
    LinExpr
        A LinExpr wrapping the constant.
    """
    const = create_const(value, size)
    return LinExpr([const], size)

def is_constant(term):
    """Is the LinOp term a constant?
    """
    return term.var_id is lo.CONSTANT_ID

def sum_expr(expressions):
    """Add linear expressions.

    Parameters
    ----------
    expression : list
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
    term : LinOp
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
    expr : LinExpr
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
    constant : LinOp
        The constant to multiply by.
    term : LinOp
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

def mul_by_const(constant, expr, size):
    """Multiply an expression on the left by a constant.

    Parameters
    ----------
    constant : LinOp
        A constant or parameter.
    expr : LinExpr
        A linear expression.
    size : tuple
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
        var_expr = LinExpr([new_var], expr.size)
        constr = create_eq(expr, var_expr, expr.size)
        constraints.append(constr)
    return (LinExpr(new_terms, size), constraints)

def mul_expr(lh_expr, rh_expr, size):
    """Multiply two expression.

    Assumes the lh_expr only contains constants.

    Parameters
    ----------
    lh_expr : LinExpr
        The left-hand expression in the product.
    rh_expr : LinExpr
        The right-hand expression in the product.
    size : tuple
        The size of the product.

    Returns
    -------
    tuple
        (LinExpr for product, list of constraints)
    """
    prod = rh_expr
    constraints = []
    for const in lh_expr.terms:
        prod, constr = mul_by_const(const, prod, size)
        constraints += constr
    return (prod, constraints)

def apply_to_vars(type_, expr, size, data=None):
    """Replaces EYE_MUL terms with terms of the given type.

    Parameters
    ----------
    type_ : str
        The type of operation to apply.
    expr : LinExpr
        The expression to apply the operation to.
    size : tuple
        The size of the expression after applying the operation.
    data :
        data for the operation.

    Returns
    -------
    tuple
        (LinExpr, list of constraints)
    """
    # Apply op if all terms are raw variables.
    if all([t.type == lo.EYE_MUL for t in expr.terms]):
        new_terms = []
        for term in expr.terms:
            new_terms.append(
                lo.LinOp(type_,
                         term.var_id,
                         term.var_size,
                         term.scalar_coeff,
                         data)
            )
        return (LinExpr(new_terms, size), [])
    # Create a constraint otherwise.
    else:
        new_var = create_var(expr.size)
        constr = create_eq(LinExpr([new_var], expr.size), expr)
        new_term = lo.LinOp(type_,
                            new_var.var_id,
                            new_var.var_size,
                            new_var.scalar_coeff,
                            data)
        return (LinExpr([new_term], size), [constr])

def sum_entries(expr):
    """Sum the entries of an expression.

    Parameters
    ----------
    expr : LinExpr
        The expression to sum the entries of.

    Returns
    -------
    tuple
        (LinExpr for sum, list of constraints)
    """
    return apply_to_vars(lo.SUM_ENTRIES, expr, (1, 1))

def index(expr, size, keys):
    """Indexes/slices an expression.

    Parameters
    ----------
    expr : LinExpr
        The expression to index.
    keys : tuple
        (row slice, column slice)
    size : tuple
        The size of the expression after indexing.

    Returns
    -------
    tuple
        (LinExpr for index, list of constraints)
    """
    return apply_to_vars(lo.INDEX, expr, size, keys)

def transpose(expr):
    """Transposes an expression.

    Parameters
    ----------
    expr : LinExpr
        The expression to transpose.

    Returns
    -------
    tuple
        (LinExpr for transpose, list of constraints)
    """
    size = (expr.size[1], expr.size[0])
    return apply_to_vars(lo.TRANSPOSE, expr, size)

def get_constr_expr(lh_expr, rh_expr):
    """Returns the expression in the constraint.
    """
    # rh_expr defaults to 0.
    if rh_expr is None:
        return lh_expr
    else:
        return sum_expr([lh_expr, neg_expr(rh_expr)])

def create_eq(lh_expr, rh_expr=None, constr_id=None):
    """Creates an internal equality constraint.

    Parameters
    ----------
    lh_term : LinExpr
        The left-hand expression in the equality constraint.
    rh_term : LinExpr
        The right-hand expression in the equality constraint.
    constr_id : int
        The id of the CVXPY equality constraint creating the constraint.

    Returns
    -------
    LinEqConstr
    """
    if constr_id is None:
        constr_id = get_id()
    expr = get_constr_expr(lh_expr, rh_expr)
    return LinEqConstr(expr, constr_id, lh_expr.size)

def create_leq(lh_expr, rh_expr=None, constr_id=None):
    """Creates an internal less than or equal constraint.

    Parameters
    ----------
    lh_term : LinExpr
        The left-hand expression in the <= constraint.
    rh_term : LinExpr
        The right-hand expression in the <= constraint.
    constr_id : int
        The id of the CVXPY equality constraint creating the constraint.

    Returns
    -------
    LinEqConstr
    """
    if constr_id is None:
        constr_id = get_id()
    expr = get_constr_expr(lh_expr, rh_expr)
    return LinLeqConstr(expr, constr_id, lh_expr.size)

def get_expr_vars(expr):
    """Get a list of the unique variables in the expression and their sizes.

    Parameters
    ----------
    expr : LinExpr
        The expression to extract the variables from.

    Returns
    -------
    list
        A list of (var id, var size) pairs, where each var id is unique.
    """
    vars_ = set()
    for term in expr.terms:
        if not is_constant(term):
            vars_.add((term.var_id, term.var_size))
    return list(vars_)
