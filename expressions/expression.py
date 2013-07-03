import settings
from coefficients import Coeff
from shape import Shape
from constraints.constraint import Constraint

class Expression(object):
    """
    A mathematical expression in a convex optimization problem.
    All attributes are functions that recurse on subexpressions.
    name - string representation of the expression.
    coefficients - Coefficients object, with dict of terms with coefficients.
    shape - Shape object, with dimensions of the evaluated expression.
    variables - A dict of name to variable containing all 
                the variables in the expression.
    subexpressions - child expressions.
    TODO priority
    """
    def __init__(self, name, coefficients, shape, variables, subexpressions):
        self.name = name
        self.coefficients = coefficients
        self.shape = shape
        self.variables = variables
        self.subexpressions = subexpressions

    def __repr__(self):
        return self.name()

    # Cast to Parameter if not an Expression.
    @staticmethod
    def const_to_param(expr):
        return expr if isinstance(expr, Expression) else Parameter(expr)

    # Helper for all binary arithmetic operators.
    # op_name - string representation of operator.
    # op_func - operator function name.
    def binary_op(self, other, op_name, op_func):
        other = Expression.const_to_param(other)
        name = lambda: ' '.join([self.name(), op_name, other.name()])
        return Expression(name,
                          lambda: getattr(self.coefficients(), op_func)(other.coefficients()),
                          lambda: getattr(self.shape(), op_func)(other.shape(),name()),
                          lambda: dict(self.variables().items() + other.variables().items()),
                          [self, other])

    # Helper for all unary arithmetic operators.
    # op_name - string representation of operator.
    # op_func - operator function name.
    def unary_op(self, op_name, op_func):
        return Expression(lambda: op_name + self.name(),
                          lambda: getattr(self.coefficients(), op_func)(),
                          lambda: getattr(self.shape(), op_func)(),
                          self.variables,
                          [self])

    """ Arithmetic operators """
    def __add__(self, other):
        return self.binary_op(other, settings.PLUS, "__add__")

    # Called for Number + Expression.
    def __radd__(self, other):
        return Parameter(other) + self

    def __sub__(self, other):
        return self.binary_op(other, settings.MINUS, "__sub__")

    # Called for Number - Expression.
    def __rsub__(self, other):
        return Parameter(other) - self

    def __mul__(self, other):
        return self.binary_op(other, settings.MUL, "__mul__") 

    # Called for Number * Expression.
    def __rmul__(self, other):
        return Parameter(other) * self

    def __neg__(self):
        return self.unary_op(settings.MINUS, "__neg__")

    """ Comparison operators """
    def __eq__(self, other):
        return Constraint(self, other, settings.EQ_CONSTR)

    def __le__(self, other):
        return Constraint(self, other, settings.INEQ_CONSTR)

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
        return Coeff({settings.CONSTANT:self.value})

    def shape(self):
        try:
            rows = len(self.value)
            try: # Matrix
                cols = len(self.value[0])
            except Exception, e: # Vector
                cols = 1
        except Exception, e: # Scalar
            rows,cols = (1,1)
        return Shape(rows,cols)

    def variables(self):
        return {}