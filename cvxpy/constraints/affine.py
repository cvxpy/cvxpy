import abc
import cvxpy.expressions.types as types
import cvxpy.utilities as u
import cvxpy.interface.matrix_utilities as intf
from cvxpy.expressions.affine import AffObjective

class AffineConstraint(u.Affine):
    """ An affine constraint. The result of canonicalization. """
    __metaclass__ = abc.ABCMeta
    def __init__(self, lh_exp, rh_exp, 
                 value_matrix=intf.DENSE_TARGET, parent=None):
        self.lh_exp = self.cast_as_affine(lh_exp)
        self.rh_exp = self.cast_as_affine(rh_exp)
        self._expr = self.lh_exp - self.rh_exp
        self.parent = parent
        self.interface = intf.get_matrix_interface(value_matrix)
        super(AffineConstraint, self).__init__()

    def name(self):
        return ' '.join([self.lh_exp.name(), 
                         self.OP_NAME, 
                         self.rh_exp.name()])

    def __str__(self):
        return self.name()

    def __repr__(self):
        return self.name()

    @property
    def size(self):
        return self._expr.size

    def variables(self):
        return self._expr.variables()

    def coefficients(self, interface):
        return self._expr.coefficients(interface)

    # Save the value of the dual variable for the constraint's parent.
    def save_value(self, value):
        if self.parent is not None:
            self.parent.dual_value = value

class AffEqConstraint(AffineConstraint):
    """ An affine equality constraint. """
    OP_NAME = "=="

class AffLeqConstraint(AffineConstraint):
    """ An affine less than or equal constraint. """
    OP_NAME = "<="