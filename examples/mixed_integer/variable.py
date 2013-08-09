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
        p_ind = sorted(cvxpy.Constant(self.value + self.u.value), 
            key=lambda pi: -abs(pi.value)) # TODO add expression value.
        self.z.value = cvxopt.matrix(0, self.size, tc='d')
        for pi in p_ind[0:self._max_card]:
            self.z.value[pi.key[0],pi.key[1]] = pi.value

    # Update step in ADMM
    def update(self):
        self.u.value += self.value - self.z.value

    # Fix the sparsity pattern by returning a constraint.
    def fix(self):
        z_ind = sorted(self.z, key=lambda zi: -abs(zi.value))
        constraints = []
        for zi in z_ind[self._max_card:]:
            constraints.append( self[zi.key[0],zi.key[1]] == 0 )
        return constraints