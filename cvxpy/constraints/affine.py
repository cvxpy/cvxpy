"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

from .. import utilities as u
from .. import interface as intf
from ..expressions import types
from ..expressions.affine import AffObjective
import abc

class AffineConstraint(u.Affine):
    """ An affine constraint. The result of canonicalization. """
    __metaclass__ = abc.ABCMeta
    def __init__(self, lh_exp, rh_exp, parent=None):
        self.lh_exp = self.cast_as_affine(lh_exp)
        self.rh_exp = self.cast_as_affine(rh_exp)
        self._expr = self.lh_exp - self.rh_exp
        self.parent = parent
        self.interface = intf.DEFAULT_INTERFACE
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