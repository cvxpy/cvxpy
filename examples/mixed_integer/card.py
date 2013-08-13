from noncvx_variable import NonCvxVariable
import cvxpy.interface.matrix_utilities as intf

class card(NonCvxVariable):
    """ A variable with constrained cardinality. """
    # k - the maximum cardinality of the variable.
    def __init__(self, rows=1, cols=1, k=None, *args, **kwargs):
        self.k = k
        super(card, self).__init__(rows, cols, *args, **kwargs)

    # All values except k-largest (by magnitude) set to zero.
    def _round(self, matrix):
        v_ind = sorted(enumerate(matrix), key=lambda v: -abs(v[1]))
        for v in v_ind[self.k:]:
           matrix[v[0]] = 0
        return matrix

    # Constrain all entries to be zero that correspond to
    # zeros in the matrix.
    def _fix(self, matrix):
        constraints = []
        rows,cols = intf.size(matrix)
        for i in range(rows):
            for j in range(cols):
                if matrix[i,j] == 0:
                    constraints.append(self[i,j] == 0)
        return constraints