"""
Copyright 2013 Steven Diamond, 2017 Akshay Agrawal, 2017 Robin Verschueren

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

from cvxpy.problems.objective import Minimize
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.dcp2cone.atom_canonicalizers import (CANON_METHODS as
                                                           cone_canon_methods)


class Dcp2Cone(Canonicalization):
    """Reduce DCP problems to a conic form.

    This reduction takes as input (minimization) DCP problems and converts
    them into problems with affine objectives and conic constraints whose
    arguments are affine.
    """
    def accepts(self, problem):
        """A problem is accepted if it is a minimization and is DCP.
        """
        return type(problem.objective) == Minimize and problem.is_dcp()

    def apply(self, problem):
        """Converts a DCP problem to a conic form.
        """
        if not self.accepts(problem):
            raise ValueError("Cannot reduce problem to cone program")
        return Canonicalization(cone_canon_methods).apply(problem)
