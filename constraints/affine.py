import abc
import cvxpy.expressions.types as types
import cvxpy.interface.matrix_utilities as intf
from cvxpy.expressions.affine import AffineObjective

class AffineConstraint(object):
    """ An affine constraint. The result of canonicalization. """
    __metaclass__ = abc.ABCMeta
    def __init__(self, lh_exp, rh_exp):
        self.lh_exp = AffineConstraint.cast_as_affine(lh_exp)
        self.rh_exp = AffineConstraint.cast_as_affine(rh_exp)
        self._expr = self.lh_exp - self.rh_exp
        super(AffineConstraint, self).__init__()

    # Casts expression as an AffineObjective.
    @staticmethod
    def cast_as_affine(expr):
        if isinstance(expr, AffineObjective):
            return expr
        elif isinstance(expr, types.expression()):
            obj,constr = expr.canonical_form()
            if len(constr) > 0:
                raise Exception("Non-affine argument '%s'." % expr.name())
            return obj
        else:
            return AffineConstraint.cast_as_affine(types.constant()(expr))

    @property
    def size(self):
        return self._expr.size

    def variables(self):
        return self._expr.variables()

    def coefficients(self, interface):
        return self._expr.coefficients(interface)

class AffEqConstraint(AffineConstraint):
    """ An affine equality constraint. """
    def __init__(self, lh_exp, rh_exp, 
                 value_matrix=intf.DENSE_TARGET, parent=None):
        self.parent = parent
        self.interface = intf.get_matrix_interface(value_matrix)
        super(AffEqConstraint, self).__init__(lh_exp, rh_exp)

    # Save the value of the dual variable for the constraint's parent.
    def save_value(self, value):
        if self.parent is not None:
            self.parent.dual_value = value

class AffLeqConstraint(AffineConstraint):
    """ An affine less than or equal constraint. """