from typing import Tuple

import numpy as np
import scipy.sparse as sp

import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.atom import Atom
from cvxpy.expressions.variable import Variable


class SuppFuncAtom(Atom):

    def __init__(self, y, parent) -> None:
        """
        Parameters
        ----------
        y : cvxpy.expressions.expression.Expression
            Must satisfy ``y.is_affine() == True``.

        parent : cvxpy.transforms.suppfunc.SuppFunc
            The object containing data for the convex set associated with this atom.
        """
        self.id = lu.get_id()
        self.args = [Atom.cast_to_const(y)]
        self._parent = parent
        self._eta = None  # store for debugging purposes
        self._shape: Tuple[int, ...] = tuple()
        self.validate_arguments()

    def validate_arguments(self) -> None:
        if self.args[0].is_complex():
            raise ValueError("Arguments to SuppFuncAtom cannot be complex.")
        if not self.args[0].is_affine():
            raise ValueError("Arguments to SuppFuncAtom must be affine.")

    def variables(self):
        varlist = self.args[0].variables()
        return varlist

    def parameters(self):
        return []

    def constants(self):
        return []

    def shape_from_args(self) -> Tuple[int, ...]:
        return self._shape

    def sign_from_args(self) -> Tuple[bool, bool]:
        return (False, False)

    def is_nonneg(self) -> bool:
        return False

    def is_nonpos(self) -> bool:
        return False

    def is_imag(self) -> bool:
        return False

    def is_complex(self) -> bool:
        return False

    def is_atom_convex(self) -> bool:
        return True

    def is_atom_concave(self) -> bool:
        return False

    def is_atom_log_log_convex(self) -> bool:
        return False

    def is_atom_log_log_concave(self) -> bool:
        return False

    def is_atom_quasiconvex(self) -> bool:
        return True

    def is_atom_quasiconcave(self) -> bool:
        return False

    def is_incr(self, idx) -> bool:
        return False

    def is_decr(self, idx) -> bool:
        return False

    def is_convex(self) -> bool:
        # The argument "y" is restricted to being affine.
        return True

    def is_concave(self) -> bool:
        return False

    def is_log_log_convex(self) -> bool:
        return False

    def is_log_log_concave(self) -> bool:
        return False

    def is_quasiconvex(self) -> bool:
        # The argument "y" is restricted to being affine.
        return True

    def is_quasiconcave(self) -> bool:
        return False

    def _value_impl(self):
        from cvxpy.problems.objective import Maximize
        from cvxpy.problems.problem import Problem
        y_val = self.args[0].value.round(decimals=9).ravel(order='F')
        x_flat = self._parent.x.flatten()
        cons = self._parent.constraints
        if len(cons) == 0:
            dummy = Variable()
            cons = [dummy == 1]
        prob = Problem(Maximize(y_val @ x_flat), cons)
        val = prob.solve(solver='SCS', eps=1e-6)
        return val

    def _grad(self, values):
        # the implementation of _grad from log_det.py was used
        # as a reference for this implementation.
        y0 = self.args[0].value  # save for later
        y = values[0]  # ignore all other values
        self.args[0].value = y
        self._value_impl()  # dead-store
        self.args[0].value = y0  # put this value back
        gradval = self._parent.x.value
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
