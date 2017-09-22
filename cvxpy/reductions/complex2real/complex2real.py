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

from cvxpy import problems
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.expression import Expression
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions import InverseData, Solution
from cvxpy.reductions.complex2real.atom_canonicalizers import (
    CANON_METHODS as elim_cplx_methods)


class Complex2Real(Reduction):
    """Eliminates piecewise linear atoms."""

    def accepts(self, problem):
        leaves = problem.variables() + problem.parameters() + problem.constants()
        return any([l.is_complex() for l in leaves])

    def apply(self, problem):
        inverse_data = InverseData(problem)

        real_obj, imag_obj = self.canonicalize_tree(
            problem.objective)
        assert imag_obj is None

        constrs = []
        for constraint in problem.constraints:
            real_constr, imag_constr = self.canonicalize_tree(
                constraint)
            assert imag_constr is None
            inverse_data.cons_id_map.update({constraint.id:
                                             real_constr.id})
            constrs.append(real_constr)

        new_problem = problems.problem.Problem(real_obj,
                                               constrs)
        return new_problem, inverse_data

    def invert(self, solution, inverse_data):
        # Add complex component.
        pvars = {}
        for vid, var in inverse_data.id2var.items():
            if var.is_real():
                pvars[vid] = solution.primal_vars[vid]
            elif var.is_imag():
                pvars[vid] = 1j*solution.primal_vars[vid]
            elif var.is_complex():
                pvars[vid] += 1j*solution.primal_vars[vid]
        dvars = {orig_id: solution.dual_vars[vid]
                 for orig_id, vid in inverse_data.cons_id_map.items()
                 if vid in solution.dual_vars}
        return Solution(solution.status, solution.opt_val, pvars, dvars,
                        solution.attr)

    def canonicalize_tree(self, expr):
        # TODO don't copy affine expressions?
        if type(expr) == cvxtypes.partial_problem():
            return NotImplemented
        else:
            real_args = []
            imag_args = []
            for arg in expr.args:
                real_arg, imag_arg = self.canonicalize_tree(arg)
                real_args.append(real_arg)
                imag_args.append(imag_arg)
            real_out, imag_out = self.canonicalize_expr(expr, real_args, imag_args)
        return real_out, imag_out

    def canonicalize_expr(self, expr, real_args, imag_args):
        if isinstance(expr, Expression) and not expr.variables():
            # Parameterized expressions are evaluated in a subsequent
            # reduction.
            if expr.parameters():
                return NotImplemented
            # Non-parameterized expressions are evaluated immediately.
            else:
                return elim_cplx_methods[Constant](Constant(expr.value),
                                                   real_args, imag_args)
        elif type(expr) in elim_cplx_methods:
            return elim_cplx_methods[type(expr)](expr, real_args, imag_args)
        else:
            assert all([v is None for v in imag_args])
            return expr.copy(real_args), None
