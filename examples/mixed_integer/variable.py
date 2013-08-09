import cvxpy
import cvxopt

class Variable(cvxpy.Variable):
    """ A variable with integer constraints """
    def __init__(self, *args, **kwargs):
        super(Variable, self).__init__(*args, **kwargs)
        self.z = cvxpy.Parameter(*self.size)
        self.z.value = cvxopt.matrix(0, self.size, tc='d')
        self.u = cvxpy.Parameter(*self.size)
        self.u.value = cvxopt.matrix(0, self.size, tc='d')

    # Sets the maximum cardinality.
    def max_card(self, card):
        self._max_card = card
        self.noncvx = True

    # The regularization objective for ADMM
    def reg_obj(self):
        return sum(cvxpy.square(self - self.z + self.u))

    # Projection step in ADMM
    # All values except k-largest (by magnitude) set to zero.
    def project(self):
        ind_val = self.sort_by_mag(self.value + self.u.value, self.size)
        self.z.value = cvxopt.matrix(0, self.size, tc='d')
        for index,val in ind_val[0:self._max_card]:
            self.z.value[index[0],index[1]] = val

    # Update step in ADMM
    def update(self):
        self.u.value += self.value - self.z.value

    # Fix the sparsity pattern by returning a constraint.
    def fix(self):
        ind_val = self.sort_by_mag(self.z.value, self.size)
        constraints = []
        for index,val in ind_val[self._max_card:]:
            constraints.append( self[index[0],index[1]] == 0 )
        return constraints

    # Return a list of the (indices, value) for the matrix,
    # sorted by decreasing magnitude.
    @staticmethod
    def sort_by_mag(matrix, size):
        ind_val = []
        for row in range(size[0]):
            for col in range(size[1]):
                ind_val.append( ((row,col), matrix[row,col]) )
        ind_val.sort(key=lambda tup: abs(tup[1]), reverse=True)
        return ind_val