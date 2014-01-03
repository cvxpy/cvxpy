from cvxpy.expressions.variables import Variable, IndexVariable
from cvxpy.expressions.constants import Parameter
import cvxpy.constraints.affine as aff
import cvxopt

class Boolean(Variable):
    def __init__(self, rows=1, cols=1, *args, **kwargs):
        self._LB = Parameter(rows, cols)
        self._LB.value = cvxopt.matrix(0,(rows, cols), tc='d')
        self._UB = Parameter(rows, cols)
        self._UB.value = cvxopt.matrix(1,(rows, cols), tc='d')
        self._fix_values = cvxopt.matrix(False,(rows, cols))
        super(Boolean, self).__init__(rows, cols, *args, **kwargs)

    # return a scalar view into a matrix of boolean variables
    def index_object(self, key):
        return IndexBoolean(self, key)

    def round(self):
        self.LB = cvxopt.matrix(self._rounded, self.size)
        self.UB = cvxopt.matrix(self._rounded, self.size)

    def relax(self):
        # if fix_value is true, do not change LB and UB
        for i, fixed in enumerate(self.fix_values):
            if not fixed:
                self.LB[i] = 0
                self.UB[i] = 1

    def set(self, value):
        if not isinstance(value, bool): raise "Must set to boolean value"
        self.LB = cvxopt.matrix(value, self.size)
        self.UB = cvxopt.matrix(value, self.size)
        self.fix_values = cvxopt.matrix(True, self.size)

    def unset(self):
        self.fix_values = cvxopt.matrix(False, self.size)

    @property
    def _rounded(self):
        # WARNING: attempts to access self.value
        if self.size == (1,1): return round(self.value)
        else: return [round(v) for v in self.value]

    @property
    def LB(self):
        return self._LB.value

    @LB.setter
    def LB(self, value):
        self._LB.value = value

    @property
    def UB(self):
        return self._UB.value

    @UB.setter
    def UB(self, value):
        self._UB.value = value

    @property
    def fix_values(self):
        return self._fix_values

    @fix_values.setter
    def fix_values(self, value):
        self._fix_values = value

class IndexBoolean(IndexVariable, Boolean):
    def __init__(self, parent, key):
        super(IndexBoolean, self).__init__(parent, key)
        self._LB = self.parent._LB[self.key]
        self._UB = self.parent._UB[self.key]

    def relax(self):
        if not self.fix_values:
            self.LB = 0
            self.UB = 1

    @property
    def LB(self):
        return self.parent._LB.value[self.key]

    @LB.setter
    def LB(self, value):
        self.parent._LB.value[self.key] = value

    @property
    def UB(self):
        return self.parent._UB.value[self.key]

    @UB.setter
    def UB(self, value):
        self.parent._UB.value[self.key] = value

    @property
    def fix_values(self):
        return self.parent._fix_values[self.key]

    @fix_values.setter
    def fix_values(self, value):
        self.parent._fix_values[self.key] = value
