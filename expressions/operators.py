import expression

class BinaryOperator(object):
    """
    Base class for expressions involving binary operators.
    """
    def __init__(self, lh_exp, rh_exp):
        self.lh_exp = lh_exp
        self.rh_exp = expression.Expression.cast_to_const(rh_exp)
        self.validate()

    # Test for incompatible dimensions and multiplication
    # by a non-constant on the left.
    def validate(self):
        self.size

    def name(self):
        return ' '.join([self.lh_exp.name(), 
                         self.OP_NAME, 
                         self.rh_exp.name()])

    # Is the expression a scalar constant?
    @staticmethod
    def is_scalar_consant(expr):
        return expr.curvature.is_constant() and expr.size == (1,1)

    # Returns the size of the expression if scalar constants were promoted.
    # Returns None if neither the lefthand nor righthand expressions can be
    # promoted.
    def promoted_size(self):
        if self.is_scalar_consant(self.rh_exp):
            return self.lh_exp.size
        elif self.is_scalar_consant(self.lh_exp):
            return self.rh_exp.size
        else:
            return None

    # The expression's sizes must match unless one is a scalar,
    # in which case it is promoted to the size of the other.
    @property
    def size(self):
        size = self.promoted_size()
        if size is not None:
            return size
        elif self.rh_exp.size == self.lh_exp.size:
            return self.lh_exp.size   
        else:
            raise Exception("Incompatible dimensions.")

    # Apply the appropriate arithmetic operator to the 
    # left hand and right hand curvatures.
    @property
    def curvature(self):
        return getattr(self.lh_exp.curvature, self.OP_FUNC)(self.rh_exp.curvature)

    # Canonicalize both sides, concatenate the constraints,
    # and apply the appropriate arithmetic operator to
    # the two objectives.
    def canonicalize(self):
        lh_obj,lh_constraints = self.lh_exp.canonicalize()
        rh_obj,rh_constraints = self.rh_exp.canonicalize()
        obj = getattr(lh_obj, self.OP_FUNC)(rh_obj)
        return (obj,lh_constraints + rh_constraints)

    def terms(self):
        return self.rh_exp.terms() + self.lh_exp.terms()

class UnaryOperator(object):
    """
    Base class for expressions involving unary operators. 
    """
    def __init__(self, expr):
        self.expr = expr

    def name(self):
        return self.OP_NAME + self.expr.name()
    
    def terms(self):
        return self.expr.terms()

    @property
    def size(self):
        return self.expr.size

    @property
    def curvature(self):
        return getattr(self.expr.curvature, self.OP_FUNC)()

    def canonicalize(self):
        obj,constraints = self.expr.canonicalize()
        obj = getattr(obj, self.OP_FUNC)()
        return (obj,constraints)