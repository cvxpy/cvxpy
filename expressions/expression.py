import abc
import cvxpy.settings as s
import cvxpy.interface.matrix_utilities as intf
import cvxpy.constraints.constraint as c
from operators import BinaryOperator, UnaryOperator
from curvature import Curvature

class Expression(object):
    """
    A mathematical expression in a convex optimization problem.
    All attributes are functions that recurse on subexpressions.
    """
    __metaclass__ = abc.ABCMeta
    # TODO priority
    def __repr__(self):
        return self.name()

    # Returns string representation of the expression.
    @abc.abstractmethod
    def name(self):
        return NotImplemented

    # Returns a dict of term name to coefficient in the
    # expression and a set of possible expression sizes.
    # interface - the matrix interface to convert constants
    #             into a matrix of the target class.
    def coefficients(self, interface):
        return NotImplemented

    # Returns a dict of name to variable containing all the
    # variables in the expression.
    def variables(self):
        return NotImplemented

    # Returns the dimensions of the expression.
    @abc.abstractmethod
    def size(self):
        return NotImplemented

    # Returns the curvature of the expression.
    @abc.abstractmethod
    def curvature(self):
        return NotImplemented

    # Returns an affine expression and affine constraints
    # representing the expression, creating new variables if necessary.
    @abc.abstractmethod
    def canonicalize(self):
        return NotImplemented

    # Cast to Parameter if not an Expression.
    @staticmethod
    def cast_to_const(expr):
        return expr if isinstance(expr, Expression) else Parameter(expr)

    # Get the coefficient of the constant in the expression.
    @staticmethod 
    def constant(coeff_dict):
        return coeff_dict.get(s.CONSTANT, 0)

    """ Arithmetic operators """
    def __add__(self, other):
        return AddExpression(self, other)

    # Called for Number + Expression.
    def __radd__(self, other):
        return Parameter(other) + self

    def __sub__(self, other):
        return SubExpression(self, other)

    # Called for Number - Expression.
    def __rsub__(self, other):
        return Parameter(other) - self

    def __mul__(self, other):
        return MulExpression(self, other)

    # Called for Number * Expression.
    def __rmul__(self, other):
        return Parameter(other) * self

    def __neg__(self):
        return NegExpression(self)

    """ Comparison operators """
    def __eq__(self, other):
        return c.EqConstraint(self, other)

    def __le__(self, other):
        return c.LeqConstraint(self, other)

    def __ge__(self, other):
        return Expression.cast_to_const(other) <= self


class AddExpression(BinaryOperator, Expression):
    OP_NAME = "+"
    OP_FUNC = "__add__"
    # Evaluates the left hand and right hand expressions and sums the dicts.
    def coefficients(self, interface):
        lh = self.lh_exp.coefficients(interface)
        rh = self.rh_exp.coefficients(interface)
        # got this nice piece of code off stackoverflow http://stackoverflow.com/questions/1031199/adding-dictionaries-in-python
        return dict( (n, lh.get(n, 0) + rh.get(n, 0)) for n in set(lh)|set(rh) )

class SubExpression(BinaryOperator, Expression):
    OP_NAME = "-"
    OP_FUNC = "__sub__"
    def coefficients(self, interface):
        return (self.lh_exp + -self.rh_exp).coefficients(interface)

class MulExpression(BinaryOperator, Expression):
    OP_NAME = "*"
    OP_FUNC = "__mul__"
    # Evaluates the left hand and right hand expressions,
    # checks the left hand expression is constant,
    # and multiplies all the right hand coefficients by the left hand constant.
    def coefficients(self, interface):
        if not self.lh_exp.curvature().is_constant():
            raise Exception("Cannot multiply on the left by a non-constant.")
        lh_coeff = self.lh_exp.coefficients(interface)
        rh_coeff = self.rh_exp.coefficients(interface)
        return dict((k,lh_coeff[s.CONSTANT]*v) for k,v in rh_coeff.items())

    # TODO scalar by vector/matrix
    def size(self):
        size = self.promoted_size()
        if size is not None:
            return size
        else:
            rh_rows,rh_cols = self.rh_exp.size()
            lh_rows,lh_cols = self.lh_exp.size()
            if lh_cols == rh_rows:
                return (lh_rows,rh_cols)
            else:
                raise Exception("'%s' has incompatible dimensions." % self.name())

    # Flips the curvature if the left hand expression is a negative scalar.
    # TODO is_constant instead of isinstance(...,Parameter) using Sign
    def curvature(self):
        curvature = super(MulExpression, self).curvature()
        if isinstance(self.lh_exp, Parameter) and \
           intf.is_scalar(self.lh_exp.value) and \
           intf.scalar_value(self.lh_exp.value) < 0:
           return -curvature
        else:
            return curvature

class NegExpression(UnaryOperator, Expression):
    OP_NAME = "-"
    OP_FUNC = "__neg__"
    # Negate all coefficients.
    def coefficients(self, interface):
        return dict((k,-v) for k,v in self.expr.coefficients(interface).items())

# class IndexExpression(Expression):
#     # key - a tuple of integers.
#     def __init__(self, expr, key):
#         self.expr = expr
#         self.key = key

#     def name(self):
#         return "%s[%s,%s]" % (self.expr.name(), self.key[0], self.key[1])

#     # TODO slices
#     def size(self):
#         return (1,1)

#     # Raise an Exception if the key is not a valid slice.
#     def validate_key(self):
#         rows,cols = self.expr.size()
#         if not (0 <= self.key[0] and self.key[0] < rows and \
#                 0 <= self.key[1] and self.key[1] < cols): 
#            raise Exception("Invalid indices %s,%s for '%s'." % 
#                 (self.key[0], self.key[1], self.expr.name()))

#     # TODO what happens to vectors/matrices of expressions?
#     def curvature(self):
#         return self.expr.curvature()

#     # TODO right place to error check?
#     def canonicalize(self):
#         self.validate_key()
#         return (None, [])


class Parameter(Expression):
    """
    A constant, either matrix or scalar.
    """
    def __init__(self, value, name=None):
        self.value = value
        self.param_name = name

    def name(self):
        return str(self.value) if self.param_name is None else self.param_name

    def coefficients(self, interface):
        return {s.CONSTANT: interface.const_to_matrix(self.value)}

    def variables(self):
        return {}

    def size(self):
        return intf.size(self.value)

    def curvature(self):
        return Curvature.CONSTANT

    def canonicalize(self):
        return (self,[])