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

from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.qp2quad_form.atom_canonicalizers import CANON_METHODS as qp_canon_methods


class Qp2SymbolicQp(Canonicalization):
    """
    Reduces a quadratic problem to a problem that consists of affine expressions
    and symbolic quadratic forms.
    """

    def accepts(self, problem):
        return problem.is_qp()

    def apply(self, problem):
        if not self.accepts(problem):
            raise ValueError("Cannot reduce problem to symbolic QP")
        return Canonicalization(qp_canon_methods).apply(problem)
