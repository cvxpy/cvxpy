import settings as s
import interface.matrices as intf
from coefficients import Coeff
from constraints.constraint import Constraint

class Expression(object):
    """
    A mathematical expression in a convex optimization problem.
    All attributes are functions that recurse on subexpressions.
    """
    # name - Returns string representation of the expression.
    # coefficients - Returns a dict of term name to coefficient in the
    #                expression and a set of possible expression shapes.
    # variables - Returns a dict of name to variable containing all 
    #             the variables in the expression.
    # TODO priority
    def __init__(self, name, coefficients, variables):
        self.name = name
        self.coefficients = coefficients
        self.variables = variables

    def __repr__(self):
        return self.name()

    # Cast to Parameter if not an Expression.
    @staticmethod
    def const_to_param(expr):
        return expr if isinstance(expr, Expression) else Parameter(expr)

    # Helper for all binary arithmetic operators.
    # op_name - string representation of operator.
    # coefficients - function for coefficients.
    def binary_op(self, other, op_name, coefficients):
        other = Expression.const_to_param(other)
        return Expression(lambda: ' '.join([self.name(), op_name, other.name()]),
                          coefficients,
                          lambda: dict(self.variables().items() + other.variables().items()))

    """ Arithmetic operators """
    def __add__(self, other):
        return self.binary_op(other, s.PLUS, lambda: Coeff.add(self, other))

    # Called for Number + Expression.
    def __radd__(self, other):
        return Parameter(other) + self

    def __sub__(self, other):
        return self.binary_op(other, s.MINUS, (self + -other).coefficients)

    # Called for Number - Expression.
    def __rsub__(self, other):
        return Parameter(other) - self

    def __mul__(self, other):
        return self.binary_op(other, s.MUL, lambda: Coeff.mul(self, other))

    # Called for Number * Expression.
    def __rmul__(self, other):
        return Parameter(other) * self

    def __neg__(self):
        return Expression(lambda: op_name + self.name(),
                          lambda: (Parameter(-1)*self).coefficients(),
                          self.variables)

    """ Comparison operators """
    def __eq__(self, other):
        return Constraint(self, other, s.EQ_CONSTR)

    def __le__(self, other):
        return Constraint(self, other, s.INEQ_CONSTR)

    def __ge__(self, other):
        return Expression.const_to_param(other) <= self

class Parameter(Expression):
    """
    A constant, either matrix or scalar.
    """
    def __init__(self, value, name=None):
        self.value = value
        self.param_name = name

    def name(self):
        return str(self.value) if self.param_name is None else self.param_name

    def coefficients(self):
        mat,shapes intf.const_to_matrix(self.value, intf.TARGET_MATRIX)
        return ({s.CONSTANT: mat}, shapes)

    def variables(self):
        return {}