"""
Copyright 2017 Robin Verschueren, 2017 Akshay Agrawal

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

import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, NonNeg, PowCone3D, PowConeND, SvecPSD, Zero
from cvxpy.reductions.cone_format import format_cone_prog
from cvxpy.reductions.cvx_attr2constr import convex_attributes
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.solver import Solver

# NOTE(akshayka): Small changes to this file can lead to drastic
# performance regressions. If you are making a change to this file,
# make sure to run cvxpy/tests/test_benchmarks.py to ensure that you have
# not introduced a regression.


def dims_to_solver_dict(cone_dims):
    cones = {
        'f': cone_dims.zero,
        'l': cone_dims.nonneg,
        'q': cone_dims.soc,
        'ep': cone_dims.exp,
        's': cone_dims.psd,
        'p': cone_dims.p3d,
        'pnd': cone_dims.pnd
    }
    return cones


class ConicSolver(Solver):
    """Conic solver class with reduction semantics

    ``apply`` requires a program whose constraint rows are already in this
    solver's cone layout, i.e. ``problem.formatted`` is True. The
    ``ConeFormat`` reduction establishes that: ``_build_solving_chain``
    appends it before every non-QP solver. Interfaces must not re-derive it.
    """
    # The key that maps to ConeDims in the data returned by apply().
    DIMS = "dims"

    # Every conic solver must support Zero and NonNeg constraints.
    SUPPORTED_CONSTRAINTS = [Zero, NonNeg]

    # Some solvers cannot solve problems that do not have constraints.
    # For such solvers, REQUIRES_CONSTR should be set to True.
    REQUIRES_CONSTR = False

    # If a solver supports exponential cones, it must specify the corresponding order
    # The cvxpy standard for the exponential cone is:
    #     K_e = closure{(x,y,z) |  z >= y * exp(x/y), y>0}.
    # Whenever a solver uses this convention, EXP_CONE_ORDER should be [0, 1, 2].
    EXP_CONE_ORDER = None

    def accepts(self, problem):
        return (isinstance(problem, ParamConeProg)
                and (self.MIP_CAPABLE or not problem.is_mixed_integer())
                and not convex_attributes([problem.x])
                and (len(problem.constraints) > 0 or not self.REQUIRES_CONSTR)
                and all(type(c) in self.SUPPORTED_CONSTRAINTS for c in
                        problem.constraints))

    @classmethod
    def format_constraints(cls, problem, exp_cone_order):
        """Deprecated: the ``ConeFormat`` chain step applies the cone layout.

        Kept because out-of-tree interfaces still call this at the top of their
        own ``apply``. On the already-formatted program they now receive it is
        a no-op, so such an interface keeps working unchanged.
        """
        return format_cone_prog(problem, exp_cone_order)

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = solution['status']

        if status in s.SOLUTION_PRESENT:
            opt_val = solution['value'] + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[self.VAR_ID]: solution['primal']}
            eq_dual = utilities.get_dual_values(
                solution['eq_dual'],
                utilities.extract_dual_value,
                inverse_data[Solver.EQ_CONSTR])
            leq_dual = utilities.get_dual_values(
                solution['ineq_dual'],
                utilities.extract_dual_value,
                inverse_data[Solver.NEQ_CONSTR])
            eq_dual.update(leq_dual)
            dual_vars = eq_dual
            return Solution(status, opt_val, primal_vars, dual_vars, {})
        else:
            return failure_solution(status)

    def _prepare_data_and_inv_data(self, problem):
        data = {}
        inv_data = {self.VAR_ID: problem.x.id}

        # Rows arrive in this order, established by the ConeFormat reduction.
        # By default cvxpy follows the SCS convention, which requires
        # constraints to be specified in the following order:
        # 1. zero cone
        # 2. non-negative orthant
        # 3. soc
        # 4. psd
        # 5. exponential
        # 6. three-dimensional power cones
        # 7. n-dimensional power cones
        data[s.PARAM_PROB] = problem
        data[self.DIMS] = problem.cone_dims
        inv_data[self.DIMS] = problem.cone_dims

        constr_map = problem.constr_map
        inv_data[self.EQ_CONSTR] = constr_map[Zero]
        inv_data[self.NEQ_CONSTR] = constr_map[NonNeg] + constr_map[SOC] + \
            constr_map.get(PSD, []) + constr_map.get(SvecPSD, []) + \
            constr_map[ExpCone] + \
            constr_map[PowCone3D] + \
            constr_map[PowConeND]
        return problem, data, inv_data

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """

        # This is a reference implementation following SCS conventions
        # Implementations for other solvers may amend or override the implementation entirely

        problem, data, inv_data = self._prepare_data_and_inv_data(problem)

        # Apply parameter values.
        # Obtain A, b such that Ax + s = b, s \in cones.
        if problem.P is None:
            c, d, A, b = problem.apply_parameters()
        else:
            P, c, d, A, b = problem.apply_parameters(quad_obj=True)
            data[s.P] = P
        data[s.C] = c
        inv_data[s.OFFSET] = d
        data[s.A] = -A
        data[s.B] = b
        data[s.LOWER_BOUNDS] = problem.lower_bounds
        data[s.UPPER_BOUNDS] = problem.upper_bounds
        return data, inv_data
