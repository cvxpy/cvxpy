import cvxpy
import cvxopt

class Variable(cvxpy.Variable):
    """ A variable with integer constraints """
    def __init__(self, *args, **kwargs):
        super(Variable, self).__init__(*args, **kwargs)
        self.noncvx = False
        self.z = Parameter(*self.size)
        self.z.value = cvxopt.matrix(0, self.size)
        self.u = Parameter(*self.size)
        self.u.value = cvxopt.matrix(0, self.size)

    # Sets the maximum cardinality.
    def max_card(self, card):
        self._max_card = card
        self.noncvx = True

    # The regularization objective for ADMM
    def reg_obj(self):
        return sum(cvxpy.square(self - self.z + self.u))

    # Projection step in ADMM
    def project(self):
        # Find k-largest values.
        ind_val = []
        for row in self.size[0]:
            for col in self.size[1]:
                ind_val.append( ((row,col), 
                    self.value[row,col] + self.u.value[row,col])
                )
        ind_val.sort(key=lambda tup: abs(tup[1]), reverse=True)
        # All values set to 0 except for _max_card largest
        # magnitudes.
        self.z.value = cvxopt.matrix(0, self.size)
        for i in range(self._max_card):
            index,val = ind_val[i]
            self.z.value[*index] = val

    # Update step in ADMM
    def update(self):
        self.u.value += self.value - self.z.value