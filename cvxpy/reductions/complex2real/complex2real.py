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
from cvxpy import settings as s
from cvxpy.atoms.affine.upper_tri import vec_to_upper_tri
from cvxpy.constraints import (
    PSD,
    SOC,
    Equality,
    Inequality,
    NonNeg,
    NonPos,
    OpRelEntrConeQuad,
    Zero,
)
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions import cvxtypes
from cvxpy.lin_ops import lin_utils as lu
from cvxpy.reductions import InverseData, Solution
from cvxpy.reductions.complex2real.canonicalizers import CANON_METHODS as elim_cplx_methods
from cvxpy.reductions.reduction import Reduction


def accepts(problem) -> bool:
    leaves = problem.variables() + problem.parameters() + problem.constants()
    return any(leaf.is_complex() for leaf in leaves)


class Complex2Real(Reduction):
    """Lifts complex numbers to a real representation."""

    UNIMPLEMENTED_COMPLEX_DUALS = (SOC, OpRelEntrConeQuad)

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
            if isinstance(real_constrs, list):
                constrs.extend(real_constrs)
            elif isinstance(real_constrs, Constraint):
                constrs.append(real_constrs)
            if isinstance(imag_constrs, list):
                constrs.extend(imag_constrs)
            elif isinstance(imag_constrs, Constraint):
                constrs.append(imag_constrs)

        new_problem = problems.problem.Problem(real_obj,
                                               constrs)
        return new_problem, inverse_data

    def invert(self, solution, inverse_data):
        pvars = {}
        dvars = {}
        if solution.status in s.SOLUTION_PRESENT:
            #
            #   Primal variables
            #
            for vid, var in inverse_data.id2var.items():
                if var.is_real():
                    # Purely real variables
                    pvars[vid] = solution.primal_vars[vid]
                elif var.is_imag():
                    # Purely imaginary variables
                    imag_id = inverse_data.real2imag[vid]
                    pvars[vid] = 1j*solution.primal_vars[imag_id]
                elif var.is_complex() and var.is_hermitian():
                    # Hermitian variables
                    pvars[vid] = solution.primal_vars[vid]
                    imag_id = inverse_data.real2imag[vid]
                    if imag_id in solution.primal_vars:
                        imag_val = solution.primal_vars[imag_id]
                        imag_val = vec_to_upper_tri(imag_val, True).value
                        imag_val -= imag_val.T
                        pvars[vid] = pvars[vid] + 1j*imag_val
                elif var.is_complex():
                    # General complex variables
                    pvars[vid] = solution.primal_vars[vid]
                    imag_id = inverse_data.real2imag[vid]
                    if imag_id in solution.primal_vars:
                        imag_val = solution.primal_vars[imag_id]
                        pvars[vid] = pvars[vid] + 1j*imag_val
            if solution.dual_vars:
                #
                #   Dual variables
                #
                for cid, cons in inverse_data.id2cons.items():
                    if cons.is_real():
                        dvars[cid] = solution.dual_vars[cid]
                    elif cons.is_imag():
                        imag_id = inverse_data.real2imag[cid]
                        dvars[cid] = 1j*solution.dual_vars[imag_id]
                    # All cases that follow are for complex-valued constraints:
                    #   1. check inequality / equality constraints.
                    #   2. check PSD constraints.
                    #   3. check if a constraint is known to lack a complex dual implementation
                    #   4. raise an error
                    elif isinstance(cons, (Equality, Zero, Inequality, NonNeg, NonPos)):
                        imag_id = inverse_data.real2imag[cid]
                        if imag_id in solution.dual_vars:
                            dvars[cid] = solution.dual_vars[cid] + \
                                1j*solution.dual_vars[imag_id]
                        else:
                            dvars[cid] = solution.dual_vars[cid]
                    elif isinstance(cons, PSD):
                        # Suppose we have a constraint con_x = X >> 0 where X is Hermitian.
                        #
                        # Define the matrix
                        #     Y := [re(X) , im(X)]
                        #          [-im(X), re(X)]
                        # and the constraint con_y = Y >> 0.
                        #
                        # The real part the dual variable for con_x is the upper-left
                        # block of the dual variable for con_y.
                        #
                        # The imaginary part of the dual variable for con_x is the
                        # upper-right block of the dual variable for con_y.
                        n = cons.args[0].shape[0]
                        dual = solution.dual_vars[cid]
                        dvars[cid] = dual[:n, :n] + 1j*dual[n:, :n]
                    elif isinstance(cons, self.UNIMPLEMENTED_COMPLEX_DUALS):
                        # TODO: implement dual variable recovery
                        pass
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
            real_out = expr.copy(real_args)
            return real_out, None
