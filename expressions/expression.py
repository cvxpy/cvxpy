import abc
import cvxpy.constraints.constraint as c
import cvxpy.settings as s
from operators import BinaryOperator, UnaryOperator
import types
import cvxpy.interface.matrix_utilities as intf
from collections import deque

class Expression(object):
    """
    A mathematical expression in a convex optimization problem.
    """
    __metaclass__ = abc.ABCMeta
    # TODO priority
    def __repr__(self):
        return self.name()

    # Returns string representation of the expression.
    @abc.abstractmethod
    def name(self):
        return NotImplemented

    # Determines the coefficients of the expression and returns
    # any parameter constraints.
    def simplify(self, interface):
        total = {}
        constraints = []
        for term,mult in self.terms():
            coeff,constr = term.dequeue_mults(mult, interface)
            constraints += constr
            total = dict( 
                         (n, total.get(n, 0) + coeff.get(n, 0)) 
                         for n in set(total)|set(coeff) 
                        )
        self.coefficients = total
        return constraints

    # Returns a dictionary of name to variable.
    # TODO necessary?
    def variables(self):
        vars = {}
        for term,mults in self.terms():
            if not term.curvature.is_constant():
                vars[term.id] = term
        return vars

    # Returns a list of (leaf, multiplication queue) for 
    # each leaf in the expression.
    def terms(self):
        return NotImplemented

    # Returns the dimensions of the expression.
    @abc.abstractproperty
    def size(self):
        return NotImplemented

    # Returns the curvature of the expression.
    @abc.abstractproperty
    def curvature(self):
        return NotImplemented

    # Returns an affine expression and affine constraints
    # representing the expression, creating new variables if necessary.
    @abc.abstractmethod
    def canonicalize(self):
        return NotImplemented

    # Cast to Constant if not an Expression.
    @staticmethod
    def cast_to_const(expr):
        return expr if isinstance(expr, Expression) else types.constant()(expr)

    # Get the coefficient of the constant in the expression.
    @staticmethod 
    def constant(coeff_dict):
        return coeff_dict.get(s.CONSTANT, 0)

    """ Arithmetic operators """
    def __add__(self, other):
        return AddExpression(self, other)

    # Called for Number + Expression.
    def __radd__(self, other):
        return Expression.cast_to_const(other) + self

    def __sub__(self, other):
        return SubExpression(self, other)

    # Called for Number - Expression.
    def __rsub__(self, other):
        return Expression.cast_to_const(other) - self

    def __mul__(self, other):
        return MulExpression(self, other)

    # Called for Number * Expression.
    def __rmul__(self, other):
        return Expression.cast_to_const(other) * self

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
    def terms(self):
        return self.lh_exp.terms() + self.rh_exp.terms()

class SubExpression(BinaryOperator, Expression):
    OP_NAME = "-"
    OP_FUNC = "__sub__"
    def terms(self):
        return (self.lh_exp + -self.rh_exp).terms()

class MulExpression(BinaryOperator, Expression):
    OP_NAME = "*"
    OP_FUNC = "__mul__"
    # Verify that left hand side is constant.
    def canonicalize(self):
        if not self.lh_exp.curvature.is_constant():
            raise Exception("Cannot multiply on the left by a non-constant.")
        return super(MulExpression, self).canonicalize()

    # Distribute left hand side across right hand side.
    # Form multiplication stacks.
    def terms(self):
        terms = []
        for rh_term,rh_mults in self.rh_exp.terms():
            for lh_term,lh_mults in self.lh_exp.terms():
                mults = deque(rh_mults)
                mults.extend(lh_mults)
                terms.append( (rh_term, mults) )
        return terms

    # TODO scalar by vector/matrix
    @property
    def size(self):
        size = self.promoted_size()
        if size is not None:
            return size
        else:
            rh_rows,rh_cols = self.rh_exp.size
            lh_rows,lh_cols = self.lh_exp.size
            if lh_cols == rh_rows:
                return (lh_rows,rh_cols)
            else:
                raise Exception("'%s' has incompatible dimensions." % self.name())

    # Flips the curvature if the left hand expression is a negative scalar.
    # TODO is_constant instead of isinstance(...,Constant) using Sign
    @property
    def curvature(self):
        curvature = super(MulExpression, self).curvature
        if isinstance(self.lh_exp, types.constant()) and \
           self.lh_exp.size == (1,1) and \
           intf.scalar_value(self.lh_exp.value) < 0:
           return -curvature
        else:
            return curvature

class NegExpression(UnaryOperator, Expression):
    OP_NAME = "-"
    OP_FUNC = "__neg__"
    # Negate all the terms.
    def terms(self):
        terms = []
        for term,mults in self.expr.terms():
            mults.append(types.constant()(-1))
            terms.append( (term, mults) )
        return terms

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
#         rows,cols = self.expr.size
#         if not (0 <= self.key[0] and self.key[0] < rows and \
#                 0 <= self.key[1] and self.key[1] < cols): 
#            raise Exception("Invalid indices %s,%s for '%s'." % 
#                 (self.key[0], self.key[1], self.expr.name()))

#     # TODO what happens to vectors/matrices of expressions?
#     def curvature(self):
#         return self.expr.curvature

#     # TODO right place to error check?
#     def canonicalize(self):
#         self.validate_key()
#         return (None, [])