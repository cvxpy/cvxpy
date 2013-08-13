from noncvx_variable import NonCvxVariable
from cvxpy.constraints.affine import AffLeqConstraint

class integer(NonCvxVariable):
    """ An integer variable. """
    # All values set rounded to the nearest integer.
    def _round(self, matrix):
        for i,v in enumerate(matrix):
            matrix[i] = round(v)
        return matrix

    # Constrain all entries to be the value in the matrix.
    def _fix(self, matrix):
        return [self == matrix]