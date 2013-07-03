import settings

class Constraint(object):
    """
    A constraint on an optimization problem of the form
    affine == affine or affine <= affine.
    Stored internally as affine <=/== 0.
    lhs - the expression on the left hand side of the constraint.
    rhs - the expression on the right hand side of the constraint.
    type - the type of constraint (a string).
    """
    def __init__(self, lhs, rhs, type):
        self.lhs = lhs
        self.rhs = rhs
        self.type = type

    def __repr__(self):
        return ' '.join([self.lhs.__repr__(), 
                         self.type, 
                         self.rhs.__repr__()])

    def coefficients(self):
        return (self.lhs - self.rhs).coefficients()

    def shape(self):
        return (self.lhs - self.rhs).shape()

    def variables(self):
        return (self.lhs - self.rhs).variables()