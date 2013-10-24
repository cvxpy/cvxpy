import expression
import operator as op

class UnaryOperator(expression.Expression):
    """
    Base class for expressions involving unary operators. 
    """
    def __init__(self, expr):
        self.expr = expr
        self.subexpressions = [expr]
        self._context = self.OP_FUNC(self.expr._context)
        super(UnaryOperator, self).__init__()

    # Applies the unary operator to the value.
    def numeric(self, values):
        return self.OP_FUNC(values[0])

    def name(self):
        return self.OP_NAME + self.expr.name()

    def canonicalize(self):
        obj,constraints = self.expr.canonical_form()
        obj = self.OP_FUNC(obj)
        return (obj,constraints)

    # Apply the appropriate arithmetic operator to the expression
    # at the given index. Return the result.
    def index_object(self, key):
        return self.OP_FUNC(self.expr[key])

    # The transpose of the unary operator.
    def transpose(self):
        return self.OP_FUNC(self.expr.T)

class NegExpression(UnaryOperator):
    OP_NAME = "-"
    OP_FUNC = op.neg