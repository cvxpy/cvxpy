import expression

class BinaryOperator(object):
    """
    Base class for expressions involving binary operators.
    """
    def __init__(self, lh_exp, rh_exp):
        self.lh_exp = lh_exp
        self.rh_exp = expression.Expression.cast_to_const(rh_exp)
        super(BinaryOperator, self).__init__()

    def name(self):
        return ' '.join([self.lh_exp.name(), 
                         self.OP_NAME, 
                         self.rh_exp.name()])

class UnaryOperator(object):
    """
    Base class for expressions involving unary operators. 
    """
    def __init__(self, expr):
        self.expr = expr
        self._shape = expr._shape
        self._curvature = getattr(self.expr.curvature, self.OP_FUNC)()
        super(UnaryOperator, self).__init__()

    def name(self):
        return self.OP_NAME + self.expr.name()

    def canonicalize(self):
        obj,constraints = self.expr.canonical_form()
        obj = getattr(obj, self.OP_FUNC)()
        return (obj,constraints)

    @property
    def size(self):
        return self._shape.size

    @property
    def curvature(self):
        return self._curvature

    # Apply the appropriate arithmetic operator to the expression
    # at the given index. Return the result.
    def index_object(self, key):
        return getattr(self.expr[key], self.OP_FUNC)()