import abc
import cvxpy.constraints.constraint as c
import cvxpy.settings as s
from operators import BinaryOperator, UnaryOperator
import types
import cvxpy.interface.matrix_utilities as intf

class Expression(object):
    """
    A mathematical expression in a convex optimization problem.
    """
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        self.objective,self.constraints = self.canonicalize()
        super(Expression, self).__init__()

    # Returns the objective and a shallow copy of the constraints list.
    def canonical_form(self):
        return (self.objective,self.constraints[:])

    # Returns an affine expression and affine constraints
    # representing the expression's objective and constraints
    # as a partial optimization problem.
    # Creates new variables if necessary.
    @abc.abstractmethod
    def canonicalize(self):
        return NotImplemented

    # TODO priority
    def __repr__(self):
        return self.name()

    # Returns string representation of the expression.
    @abc.abstractmethod
    def name(self):
        return NotImplemented

    # The curvature of the expression.
    @abc.abstractproperty
    def curvature(self):
        return NotImplemented

    # The dimensions of the expression.
    @abc.abstractproperty
    def size(self):
        return NotImplemented

    # Cast to Constant if not an Expression.
    @staticmethod
    def cast_to_const(expr):
        return expr if isinstance(expr, Expression) else types.constant()(expr)

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
    def __init__(self, lh_exp, rh_exp):
        super(AddExpression, self).__init__(lh_exp, rh_exp)
        self.set_curvature()
        self.set_shape()

    @property
    def size(self):
        return self._shape.size

    def set_shape(self):
        self._shape = self.lh_exp._shape + self.rh_exp._shape

    # Apply the appropriate arithmetic operator to the 
    # left hand and right hand curvatures.
    def set_curvature(self):
        self._curvature = getattr(self.lh_exp.curvature,
                                  self.OP_FUNC)(self.rh_exp.curvature)

    @property
    def curvature(self):
        return self._curvature

    # Canonicalize both sides, concatenate the constraints,
    # and apply the appropriate arithmetic operator to
    # the two objectives.
    def canonicalize(self):
        lh_obj,lh_constraints = self.lh_exp.canonical_form()
        rh_obj,rh_constraints = self.rh_exp.canonical_form()
        obj = getattr(lh_obj, self.OP_FUNC)(rh_obj)
        return (obj,lh_constraints + rh_constraints)

class SubExpression(AddExpression, Expression):
    OP_NAME = "-"
    OP_FUNC = "__sub__"

class MulExpression(AddExpression, Expression):
    OP_NAME = "*"
    OP_FUNC = "__mul__"
    def __init__(self, lh_exp, rh_exp):
        super(MulExpression, self).__init__(lh_exp, rh_exp)
        # Left hand expression must be constant.
        if not lh_exp.curvature.is_constant():
            raise Exception("Cannot multiply on the left by a non-constant.")

    def set_shape(self):
        self._shape = self.lh_exp._shape * self.rh_exp._shape

    # Flips the curvature if the left hand expression is a negative scalar.
    # TODO is_constant instead of isinstance(...,Constant) using Sign
    def set_curvature(self):
        if isinstance(self.lh_exp, types.constant()) and \
            self.lh_exp.size == (1,1) and \
            intf.scalar_value(self.lh_exp.value) < 0:
            self._curvature = -self.rh_exp.curvature
        else:
            self._curvature = self.rh_exp.curvature

class NegExpression(UnaryOperator, Expression):
    OP_NAME = "-"
    OP_FUNC = "__neg__"