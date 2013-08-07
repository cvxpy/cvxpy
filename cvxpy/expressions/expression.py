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

    # Is the expression a scalar?
    def is_scalar(self):
        return self.size == (1,1)

    # Is the expression a column vector?
    def is_vector(self):
        return self.size[1] == 1

    # Cast to Constant if not an Expression.
    @staticmethod
    def cast_to_const(expr):
        return expr if isinstance(expr, Expression) else types.constant()(expr)

    def __coerce__(self, other):
        return (self, cast_to_const(other))

    """ Iteration """
    # Raise an Exception if the key is not a valid index.
    # Returns the key as a tuple.
    def validate_key(self, key):
        rows,cols = self.size
        # Change single indexes for vectors into double indices.
        if not isinstance(key, tuple):
            if rows == 1:
                key = (0,key)
            elif cols == 1:
                key = (key,0)
            else:
                raise Exception("Invalid index %s for '%s'." % (key, self.name()))
        # Check that index is in bounds.
        if not (0 <= key[0] and key[0] < rows and \
                0 <= key[1] and key[1] < cols):
           raise Exception("Invalid indices %s,%s for '%s'." % 
                (key[0], key[1], self.name()))
        return key

    # Create a new variable that acts as a view into this variable.
    # Updating the variable's value updates the value of this variable instead.
    def __getitem__(self, key):
        key = self.validate_key(key)
        # Indexing into a scalar returns the scalar.
        if self.size == (1,1):
            return self
        else:
            return self.index_object(key)
        # TODO # Set value if variable has value.
        # if self.value is not None:
        #     index_var.primal_value = self.value[key]

    # Iterating over the leaf returns the index expressions
    # in column major order.
    def __iter__(self):
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                yield self[row,col]

    def __len__(self):
        length = self.size[0]*self.size[1]
        if length == 1: # Numpy will iterate over anything with a length.
            return NotImplemented
        else:
            return length

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
        return Expression.cast_to_const(other).__le__(self)


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

    # Return the symbolic affine expression equal to the given index
    # into the expression.
    def index_object(self, key):
        # Scalar promotion
        promoted = self.promoted_index_object(key)
        if promoted is not None:
            return promoted
        return getattr(self.lh_exp[key], self.OP_FUNC)(self.rh_exp[key])

    # Handle promoted scalars.
    def promoted_index_object(self, key):
        if self.lh_exp.size == (1,1):
            return getattr(self.lh_exp, self.OP_FUNC)(self.rh_exp[key])
        elif self.rh_exp.size == (1,1):
            return getattr(self.lh_exp[key], self.OP_FUNC)(self.rh_exp)
        else:
            return None

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
        if not self.lh_exp.curvature.is_constant():
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

    # Return the symbolic affine expression equal to the given index
    # in the expression.
    def index_object(self, key):
        # Scalar multiplication
        promoted = self.promoted_index_object(key)
        if promoted is not None:
            return promoted
        # Matrix multiplication.
        val = 0
        for i in range(self.lh_exp.size[1]):
            val = val + self.lh_exp[key[0],i] * self.rh_exp[i,key[1]]
        return val

class NegExpression(UnaryOperator, Expression):
    OP_NAME = "-"
    OP_FUNC = "__neg__"