import expression

class UnaryOperator(expression.Expression):
    """
    Base class for expressions involving unary operators. 
    """
    def __init__(self, expr):
        self.expr = expr
        self._context = getattr(self.expr._context, self.OP_FUNC)()
        super(UnaryOperator, self).__init__()

    def name(self):
        return self.OP_NAME + self.expr.name()

    def canonicalize(self):
        obj,constraints = self.expr.canonical_form()
        obj = getattr(obj, self.OP_FUNC)()
        return (obj,constraints)

    # Apply the appropriate arithmetic operator to the expression
    # at the given index. Return the result.
    def index_object(self, key):
        return getattr(self.expr[key], self.OP_FUNC)()

    # The transpose of the unary operator.
    def transpose(self):
        return getattr(self.expr.T, self.OP_FUNC)()

class NegExpression(UnaryOperator):
    OP_NAME = "-"
    OP_FUNC = "__neg__"