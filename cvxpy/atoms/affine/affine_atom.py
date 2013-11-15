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

import abc
from ..atom import Atom
from ...import utilities as u

class AffAtom(Atom, u.Affine):
    """ Abstract base class for affine atoms. """
    __metaclass__ = abc.ABCMeta
    # The curvature of the atom if all arguments conformed to DCP.
    def func_curvature(self):
        return u.Curvature.AFFINE

    # Doesn't matter for affine atoms.
    def monotonicity(self):
        return len(self.args)*[u.monotonicity.INCREASING]

    def graph_implementation(self, arg_objs):
        # By default, canonicalization applies the atom to the arg_objs.
        return (self.__class__(*arg_objs), [])
