import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.atom import Atom
from cvxpy.expressions.variable import Variable
import scipy.sparse as sp
import numpy as np


class SuppFuncAtom(Atom):

    def __init__(self, y, A, b, K_sels, x, cons):
        self.id = lu.get_id()
        eta = Variable(shape=(b.size,))
        self.args = [Atom.cast_to_const(y), Atom.cast_to_const(A), Atom.cast_to_const(b),
                     Atom.cast_to_const(eta)]
        horrible = self.args[2]
        horrible.__dict__['K_sels'] = K_sels
        self._x = x
        self._xcons = cons
        self.validate_arguments()
        self._shape = tuple()
        pass

    def validate_arguments(self):
        if self.args[0].is_complex():
            raise ValueError("Arguments to SuppFuncAtom cannot be complex.")
        if not self.args[0].is_affine():
            raise ValueError("Arguments to SuppFuncAtom must be affine.")

    def variables(self):
        varlist = self.args[0].variables() + self.args[3].variables()
        return varlist

    def parameters(self):
        return []

    def constants(self):
        return [self.args[1], self.args[2]]

    def shape_from_args(self):
        return self._shape

    def sign_from_args(self):
        return (False, False)

    def is_nonneg(self):
        return False

    def is_nonpos(self):
        return False

    def is_imag(self):
        return False

    def is_complex(self):
        return False

    def is_atom_convex(self):
        return True

    def is_atom_concave(self):
        return False

    def is_atom_log_log_convex(self):
        return False

    def is_atom_log_log_concave(self):
        return False

    def is_atom_quasiconvex(self):
        return True

    def is_atom_quasiconcave(self):
        return False

    def is_incr(self, idx):
        return False

    def is_decr(self, idx):
        return False

    def is_convex(self):
        # The argument "y" is restricted to being affine.
        return True

    def is_concave(self):
        return False

    def is_log_log_convex(self):
        return False

    def is_log_log_concave(self):
        return False

    def is_quasiconvex(self):
        # The argument "y" is restricted to being affine.
        return True

    def is_quasiconcave(self):
        return False

    def _value_impl(self):
        from cvxpy.problems.problem import Problem
        from cvxpy.problems.objective import Maximize
        y_val = self.args[0].value.round(decimals=9).ravel(order='F')
        x_flat = self._x.flatten()
        cons = self._xcons
        prob = Problem(Maximize(y_val @ x_flat), cons)
        val = prob.solve()
        return val

    def _grad(self, values):
        # the implementation of _grad from log_det.py was used
        # as a reference for this implementation.
        y0 = self.args[0].value  # save for later
        y = values[0]  # ignore all other values
        self.args[0].value = y
        self._value_impl()  # dead-store
        self.args[0].value = y0  # put this value back
        gradval = self._x.value
        if np.any(np.isnan(gradval)):
            # If we evaluated the support function successfully, then
            # this means the support function is not finite at this input.
            return [None]
        else:
            gradmat = sp.csc_matrix(gradval.ravel(order='F')).T
            return [gradmat]

    def __lt__(self, other):
        raise NotImplementedError("Strict inequalities are not allowed.")

    def __gt__(self, other):
        raise NotImplementedError("Strict inequalities are not allowed.")
