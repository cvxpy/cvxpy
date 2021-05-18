"""
Copyright 2017 Robin Verschueren

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from cvxpy import problems
from cvxpy.expressions import cvxtypes
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions import InverseData, Solution
from cvxpy.constraints import (Equality, Inequality,
                               Zero, NonNeg,
                               PSD, SOC, NonPos)
from cvxpy.reductions.complex2real.atom_canonicalizers import (
    CANON_METHODS as elim_cplx_methods)
from cvxpy.lin_ops import lin_utils as lu
from cvxpy import settings as s


def accepts(problem) -> bool:
    leaves = problem.variables() + problem.parameters() + problem.constants()
    return any(leaf.is_complex() for leaf in leaves)


class Complex2Real(Reduction):
    """Lifts complex numbers to a real representation."""

    def accepts(self, problem) -> None:
        accepts(problem)

    def apply(self, problem):
        inverse_data = InverseData(problem)
        real2imag = {var.id: lu.get_id() for var in problem.variables()
                     if var.is_complex()}
        constr_dict = {cons.id: lu.get_id() for cons in problem.constraints
                       if cons.is_complex()}
        real2imag.update(constr_dict)
        inverse_data.real2imag = real2imag

        leaf_map = {}
        real_obj, imag_obj = self.canonicalize_tree(
            problem.objective, inverse_data.real2imag, leaf_map)
        assert imag_obj is None

        constrs = []
        for constraint in problem.constraints:
            # real2imag maps variable id to a potential new variable
            # created for the imaginary part.
            real_constrs, imag_constrs = self.canonicalize_tree(
                constraint, inverse_data.real2imag, leaf_map)
            if real_constrs is not None:
                constrs.extend(real_constrs)
            if imag_constrs is not None:
                constrs.extend(imag_constrs)

        new_problem = problems.problem.Problem(real_obj,
                                               constrs)
        return new_problem, inverse_data

    def invert(self, solution, inverse_data):
        pvars = {}
        dvars = {}
        if solution.status in s.SOLUTION_PRESENT:
            for vid, var in inverse_data.id2var.items():
                if var.is_real():
                    pvars[vid] = solution.primal_vars[vid]
                elif var.is_imag():
                    imag_id = inverse_data.real2imag[vid]
                    pvars[vid] = 1j*solution.primal_vars[imag_id]
                elif var.is_complex() and var.is_hermitian():
                    imag_id = inverse_data.real2imag[vid]
                    # Imaginary part may have been lost.
                    if imag_id in solution.primal_vars:
                        imag_val = solution.primal_vars[imag_id]
                        pvars[vid] = solution.primal_vars[vid] + \
                            1j*(imag_val - imag_val.T)/2
                    else:
                        pvars[vid] = solution.primal_vars[vid]
                elif var.is_complex():
                    imag_id = inverse_data.real2imag[vid]
                    # Imaginary part may have been lost.
                    if imag_id in solution.primal_vars:
                        pvars[vid] = solution.primal_vars[vid] + \
                            1j*solution.primal_vars[imag_id]
                    else:
                        pvars[vid] = solution.primal_vars[vid]
            for cid, cons in inverse_data.id2cons.items():
                if cons.is_real():
                    dvars[cid] = solution.dual_vars[cid]
                elif cons.is_imag():
                    imag_id = inverse_data.real2imag[cid]
                    dvars[cid] = 1j*solution.dual_vars[imag_id]
                # For equality and inequality constraints.
                elif isinstance(cons,
                                (Equality, Zero, Inequality,
                                 NonNeg, NonPos)
                                ) and cons.is_complex():
                    imag_id = inverse_data.real2imag[cid]
                    if imag_id in solution.dual_vars:
                        dvars[cid] = solution.dual_vars[cid] + \
                            1j*solution.dual_vars[imag_id]
                    else:
                        dvars[cid] = solution.dual_vars[cid]
                elif isinstance(cons, SOC) and cons.is_complex():
                    # TODO add dual variables for complex SOC.
                    pass
                # For PSD constraints.
                elif isinstance(cons, PSD) and cons.is_complex():
                    n = cons.args[0].shape[0]
                    dual = solution.dual_vars[cid]
                    dvars[cid] = dual[:n, :n] + 1j*dual[n:, :n]
                else:
                    raise Exception("Unknown constraint type.")

        return Solution(solution.status, solution.opt_val, pvars, dvars,
                        solution.attr)

    def canonicalize_tree(self, expr, real2imag, leaf_map):
        # TODO don't copy affine expressions?
        if type(expr) == cvxtypes.partial_problem():
            raise NotImplementedError()
        else:
            real_args = []
            imag_args = []
            for arg in expr.args:
                real_arg, imag_arg = self.canonicalize_tree(arg, real2imag, leaf_map)
                real_args.append(real_arg)
                imag_args.append(imag_arg)
            real_out, imag_out = self.canonicalize_expr(expr, real_args,
                                                        imag_args, real2imag,
                                                        leaf_map)
        return real_out, imag_out

    def canonicalize_expr(self, expr, real_args, imag_args, real2imag, leaf_map):
        if type(expr) in elim_cplx_methods:
            # Only canonicalize a variable/constant/parameter once.
            if len(expr.args) == 0 and expr in leaf_map:
                return leaf_map[expr]
            result = elim_cplx_methods[type(expr)](expr, real_args, imag_args, real2imag)
            if len(expr.args) == 0:
                leaf_map[expr] = result
            return result
        else:
            assert all(v is None for v in imag_args)
            return expr.copy(real_args), None
