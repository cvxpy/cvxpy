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

from cvxpy.constraints import NonPos, Zero
from cvxpy.problems.objective import Minimize
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.cvx_attr2constr import convex_attributes
from cvxpy.reductions.qp2quad_form.atom_canonicalizers import (
    CANON_METHODS as qp_canon_methods)
from cvxpy.reductions.utilities import are_args_affine


class Qp2SymbolicQp(Canonicalization):
    """
    Reduces a quadratic problem to a problem that consists of affine
    expressions and symbolic quadratic forms.
    """
    def accepts(self, problem):
        """
        Problems with quadratic, piecewise affine objectives,
        piecewise-linear constraints inequality constraints, and
        affine equality constraints are accepted.
        """
        return (((type(problem.objective) == Minimize
                  and problem.objective.expr.is_qpwa())
                 or problem.objective.expr.is_affine())
                and not set(['PSD', 'NSD']).intersection(convex_attributes(
                                                         problem.variables()))
                and all(type(c) == NonPos or type(c) == Zero
                        for c in problem.constraints)
                and all(c.expr.is_pwl() for c in problem.constraints
                        if type(c) == NonPos)
                and are_args_affine(problem.constraints))

    def apply(self, problem):
        """Converts a QP to an even more symbolic form."""
        if not self.accepts(problem):
            raise ValueError("Cannot reduce problem to symbolic QP")
        return Canonicalization(qp_canon_methods).apply(problem)
