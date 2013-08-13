from noncvx_variable import NonCvxVariable
from cvxpy.constraints.affine import AffLeqConstraint

class boolean(NonCvxVariable):
    """ A boolean variable. """
    # All values set rounded to zero or 1.
    def _round(self, matrix):
        for i,v in enumerate(matrix):
            matrix[i] = 0 if v < 0.5 else 1
        return matrix

    # Constrain all entries to be the value in the matrix.
    def _fix(self, matrix):
        return [self == matrix]

    # In the relaxation, we have 0 <= var <= 1.
    def _constraints(self):
        return [AffLeqConstraint(0, self._objective()),
                AffLeqConstraint(self._objective(), 1)]