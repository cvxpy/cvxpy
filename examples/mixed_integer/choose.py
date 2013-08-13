from noncvx_variable import NonCvxVariable
from cvxpy.constraints.affine import AffLeqConstraint, AffEqConstraint

class choose(NonCvxVariable):
    """ A variable with k 1's and all other entries 0. """
    def __init__(self, rows=1, cols=1, k=None, *args, **kwargs):
        self.k = k
        super(choose, self).__init__(rows, cols, *args, **kwargs)

    # The k-largest values are set to 1. The remainder are set to 0.
    def _round(self, matrix):
        v_ind = sorted(enumerate(matrix), key=lambda v: -v[1])
        for v in v_ind[0:self.k]:
            matrix[v[0]] = 1
        for v in v_ind[self.k:]:
            matrix[v[0]] = 0
        return matrix

    # Constrain all entries to be zero that correspond to
    # zeros in the matrix.
    def _fix(self, matrix):
        return [self == matrix]

    # In the relaxation, 0 <= var <= 1 and sum(var) == k.
    def constraints(self):
        return [AffLeqConstraint(0, self._objective()),
                AffLeqConstraint(self._objective(), 1),
                AffEqConstraint(sum(self), self.k)]