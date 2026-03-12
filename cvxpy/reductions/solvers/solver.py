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

import abc

from cvxpy import settings as s
from cvxpy.reductions.cone2cone.approx import APPROX_CONE_CONVERSIONS
from cvxpy.reductions.cone2cone.exact import EXACT_CONE_CONVERSIONS
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solvers.solver_inverse_data import SolverInverseData


def expand_cones(cones, supported):
    """Expand unsupported cones via exact and approximate conversions.

    Parameters
    ----------
    cones : set[type]
        Mutable set of cone constraint types (modified in place).
    supported : frozenset[type]
        Cones the target solver natively supports.

    Returns
    -------
    (cones, exact_targets, approx_targets) where *cones* is the
    (mutated) input set and the targets are the cones that were expanded.
    """
    exact_targets = (cones & EXACT_CONE_CONVERSIONS.keys()) - supported
    for co in exact_targets:
        cones.discard(co)
        cones.update(EXACT_CONE_CONVERSIONS[co])

    approx_targets = (cones & APPROX_CONE_CONVERSIONS.keys()) - supported
    for co in approx_targets:
        cones.discard(co)
        cones.update(APPROX_CONE_CONVERSIONS[co])

    return cones, exact_targets, approx_targets


class Solver(Reduction):
    """Generic interface for a solver that uses reduction semantics
    """

    DIMS = "dims"
    # ^ The key that maps to "ConeDims" in the data returned by apply().
    #
    #   There are separate ConeDims classes for cone programs vs QPs.
    #   See cone_matrix_stuffing.py and qp_matrix_stuffing.py for details.

    # Solver capabilities.
    MIP_CAPABLE = False
    BOUNDED_VARIABLES = False
    SOC_DIM3_ONLY = False

    # Constraint support (overridden by ConicSolver and QpSolver).
    SUPPORTED_CONSTRAINTS = []
    REQUIRES_CONSTR = False

    # Keys for inverse data.
    VAR_ID = 'var_id'
    DUAL_VAR_ID = 'dual_var_id'
    EQ_CONSTR = 'eq_constr'
    NEQ_CONSTR = 'other_constr'

    @abc.abstractmethod
    def name(self):
        """The name of the solver.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def import_solver(self):
        """Imports the solver.
        """
        raise NotImplementedError()

    def is_installed(self) -> bool:
        """Is the solver installed?
        """
        try:
            self.import_solver()
            return True
        except Exception as e:
            if not isinstance(e, ModuleNotFoundError):
                s.LOGGER.warning(
                    f"Encountered unexpected exception importing solver {self.name()}:\n"
                    + repr(e)
                )
            return False

    @abc.abstractmethod
    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        """Solve a problem represented by data returned from apply.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def cite(self, data):
        """Return bibtex citation for the solver.
        """
        raise NotImplementedError()

    def supports_quad_obj(self) -> bool:
        """Whether the solver supports quadratic objectives."""
        return False

    def can_solve(self, problem_form) -> bool:
        """Check if this solver can handle a problem with the given structure.

        Parameters
        ----------
        problem_form : ProblemForm
            Pre-canonicalization structural analysis of the problem.
        """
        if problem_form.is_mixed_integer() and not self.MIP_CAPABLE:
            return False

        if problem_form.is_mixed_integer():
            supported = frozenset(
                getattr(self, 'MI_SUPPORTED_CONSTRAINTS', self.SUPPORTED_CONSTRAINTS)
            )
        else:
            supported = frozenset(self.SUPPORTED_CONSTRAINTS)

        quad_obj = self.supports_quad_obj() and problem_form.has_quadratic_objective()
        cones = problem_form.cones(quad_obj=quad_obj).copy()
        expand_cones(cones, supported)

        if not problem_form.has_constraints() and self.REQUIRES_CONSTR:
            return False

        return cones.issubset(supported)

    def solve(self, problem, warm_start: bool, verbose: bool, solver_opts):
        """Solve the problem and return a Solution object.
        """
        data, inv_data = self.apply(problem)
        solution = self.solve_via_data(data, warm_start, verbose, solver_opts)
        inverse_data = SolverInverseData(inv_data, solver_instance=self, solver_options=solver_opts)
        return self.invert(solution, inverse_data)
