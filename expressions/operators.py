class BinaryOperator(object):
    def __init__(self, lh_exp, rh_exp):
        self.lh_exp = lh_exp
        self.rh_exp = expression.Expression.cast_to_const(rh_exp)

    def name(self):
        return ' '.join([self.lh_exp.name(), 
                         self.OP_NAME, 
                         self.rh_exp.name()])

    def variables(self):
        return dict( self.lh_exp.variables().items() + \
                     self.rh_exp.variables().items() )

    # The expression's sizes must match.
    def size(self):
        if self.rh_exp.size() != self.lh_exp.size():
            raise Exception("'%s' has incompatible dimensions." % self.name())
        return self.lh_exp.size()

    # Apply the appropriate arithmetic operator to the 
    # left hand and right hand curvatures.
    def curvature(self):
        return getattr(self.lh_exp.curvature(), self.OP_FUNC)(self.rh_exp.curvature())

    # Canonicalize both sides, concatenate the constraints,
    # and apply the appropriate arithmetic operator to
    # the two objectives.
    def canonicalize(self):
        lh_obj,lh_constraints = self.lh_exp.canonicalize()
        rh_obj,rh_constraints = self.rh_exp.canonicalize()
        obj = getattr(lh_obj, self.OP_FUNC)(rh_obj)
        return (obj,lh_constraints + rh_constraints)

class UnaryOperator(object):
    def __init__(self, expr):
        self.expr = expr

    def name(self):
        return self.OP_NAME + self.expr.name()
    
    def variables(self):
        return self.expr.variables()

    def size(self):
        return self.expr.size()

    def curvature(self):
        return getattr(self.expr.curvature(), self.OP_FUNC)()

    def canonicalize(self):
        obj,constraints = self.expr.canonicalize()
        obj = getattr(obj, self.OP_FUNC)()
        return (obj,constraints)

import expression