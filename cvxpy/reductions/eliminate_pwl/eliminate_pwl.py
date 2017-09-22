"""
Copyright 2017 Robin Verschueren

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

from cvxpy.atoms import abs, maximum, sum_largest, max, norm1, norm_inf
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.eliminate_pwl.atom_canonicalizers import (
    CANON_METHODS as elim_pwl_methods)


class EliminatePwl(Canonicalization):
    """Eliminates piecewise linear atoms."""

    def accepts(self, problem):
        atom_types = [type(atom) for atom in problem.atoms()]
        pwl_types = [abs, maximum, sum_largest, max, norm1, norm_inf]
        return any(atom in pwl_types for atom in atom_types)

    def apply(self, problem):
        if not self.accepts(problem):
            raise ValueError("Cannot canonicalize away pwl atoms.")
        return Canonicalization(elim_pwl_methods).apply(problem)
