"""
Copyright, the CVXPY authors

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
from cvxpy.constraints import PSD, NonNeg, Zero
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.solver import Solver


def dims_to_solver_dict(cone_dims):
    cones = {
        'f': cone_dims.zero,
        'l': cone_dims.nonneg,
        # 'q': cone_dims.soc,
        's': cone_dims.psd
    }
    return cones


class SDPA(ConicSolver):
    """An interface for the SDPA solver.
    """
    # Solver capabilities.
    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [PSD]

    # Map of SDPA status to CVXPY status.
    STATUS_MAP = {
        "pdOPT": s.OPTIMAL,
        "noINFO": s.SOLVER_ERROR,
        "pFEAS": s.OPTIMAL_INACCURATE,
        "dFEAS": s.OPTIMAL_INACCURATE,
        "pdFEAS": s.OPTIMAL_INACCURATE,
        "pdINF": s.INFEASIBLE,
        "pFEAS_dINF": s.UNBOUNDED,
        "pINF_dFEAS": s.INFEASIBLE,
        "pUNBD": s.UNBOUNDED,
        "dUNBD": s.INFEASIBLE  # by weak duality
    }

    def name(self):
        """The name of the solver.
        """
        return s.SDPA

    def import_solver(self) -> None:
        """Imports the solver.
        """
        import sdpap  # noqa F401

    def accepts(self, problem) -> bool:
        """Can SDPA solve the problem?
        """
        if not problem.objective.args[0].is_affine():
            return False
        for constr in problem.constraints:
            if type(constr) not in self.SUPPORTED_CONSTRAINTS:
                return False
            for arg in constr.args:
                if not arg.is_affine():
                    return False
        return True

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """

        # CVXPY represents cone programs as
        #     (P) min_x { c^T x : A x + b \in K } + d,
        # or, using Dualize
        #     (D) max_y { -b^T y : c - A^T y = 0, y \in K^* } + d.

        # SDPAP takes a conic program in CLP form:
        #     (P) min_x { c^T x : A x - b \in J, x \in K }
        # or
        #     (D) max_y { b^T y : y \in J^*, c - a^T y \in K^*}

        # SDPA takes a conic program in SeDuMI form:
        #     (P) min_x { c^T x : A x - b = 0, x \in K }
        # or
        #     (D) max_y { b^T y : c - A^T y \in K^*}

        # SDPA always takes an input in SeDuMi form
        # SDPAP therefore converts the problem from CLP form to SeDuMi form which
        # involves getting rid of constraints involving cone J (and J^*) of the CLP form

        # We have two ways to solve using SDPA

        # 1. CVXPY (P) -> CLP (P), by
        #       - flipping sign of b
        #       - setting J of CLP (P) to K of CVXPY (P)
        #       - setting K of CLP (P) to a free cone
        #
        #       (a) then CLP (P)-> SeDuMi (D) if user specifies `convMethod` option to `clp_toLMI`
        #       (b) then CLP (P)-> SeDuMi (P) if user specifies `convMethod` option to `clp_toEQ`
        #
        # 2. CVXPY (P) -> CVXPY (D), by
        #       - using Dualize
        #       - then CVXPY (D) -> SeDuMi (P) by
        #           - setting c of SeDuMi (P) to -b of CVXPY (D)
        #           - setting b of SeDuMi (P) to c of CVXPY (D)
        #           - setting x of SeDuMi (P) to y of CVXPY (D)
        #           - setting K of SeDuMi (P) to K^* of CVXPY (D)
        #           - transposing A

        # 2 does not give a benefit over 1(a)
        #   1 (a) and 2 both flip the primal and dual between CVXPY and SeDuMi
        # 1 (b) does not flip primal and dual throughout, but requires `clp_toEQ` to be implemented
        #   TODO: Implement `clp_toEQ` in `sdpa-python`
        # Thereafter, 1 will allows user to choose between solving as primal or dual
        # by specifying `convMethod` option in solver options

        data = {}
        inv_data = {self.VAR_ID: problem.x.id}

        if not problem.formatted:
            problem = self.format_constraints(problem, None)
        data[s.PARAM_PROB] = problem
        data[self.DIMS] = problem.cone_dims
        inv_data[self.DIMS] = problem.cone_dims

        constr_map = problem.constr_map
        inv_data[self.EQ_CONSTR] = constr_map[Zero]
        inv_data[self.NEQ_CONSTR] = constr_map[NonNeg] + constr_map[PSD]

        c, d, A, b = problem.apply_parameters()

        data[s.C] = c
        inv_data[s.OFFSET] = d
        data[s.A] = A
        if data[s.A].shape[0] == 0:
            data[s.A] = None
        data[s.B] = b.flatten()
        if data[s.B].shape[0] == 0:
            data[s.B] = None

        return data, inv_data

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = solution['status']

        if status in s.SOLUTION_PRESENT:
            opt_val = solution['value']
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

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        import sdpap
        from scipy import matrix

        data[s.DIMS] = dims_to_solver_dict(data[s.DIMS])

        A, b, c, dims = data[s.A], data[s.B], data[s.C], data[s.DIMS]

        # x is in the Euclidean cone (i.e. free variable) which translates to cone K in SDPAP
        # c is the same length as x
        K = sdpap.SymCone(f=c.shape[0])

        # cone K in CVXPY conic form becomes the cone J of SDPAP (after flipping the sign of b)
        J = sdpap.SymCone(f=dims['f'], l=dims['l'], s=tuple(dims['s']))

        # `solver_opts['print'] = 'display'` can override `verbose = False`.
        # User may choose to display solver output in non verbose mode.
        if 'print' not in solver_opts:
            solver_opts['print'] = 'display' if verbose else 'no'
        x, y, sdpapinfo, timeinfo, sdpainfo = sdpap.solve(
            A, -matrix(b), matrix(c), K, J, solver_opts)

        # This should be set according to the value of `#define REVERSE_PRIMAL_DUAL`
        # in `sdpa` (not `sdpa-python`) source.
        # By default it's enabled and hence, the primal problem in SDPA takes
        # the LMI form (opposite that of SeDuMi).
        # If, while building `sdpa` from source you disable it, you need to change this to `False`.
        reverse_primal_dual = True
        # By disabling `REVERSE_PRIMAL_DUAL`, the primal-dual pair
        # in SDPA becomes similar to SeDuMi, i.e. the form we want (see long note in `apply` method)
        # However, with `REVERSE_PRIMAL_DUAL` disabled, we have some
        # accuracy related issues on unit tests using the default parameters.

        REVERSE_PRIMAL_DUAL = {
            "noINFO": "noINFO",
            "pFEAS": "pFEAS",
            "dFEAS": "dFEAS",
            "pdFEAS": "pdFEAS",
            "pdINF": "pdINF",
            "pFEAS_dINF": "pINF_dFEAS",  # invert flip within sdpa
            "pINF_dFEAS": "pFEAS_dINF",  # invert flip within sdpa
            "pdOPT": "pdOPT",
            "pUNBD": "dUNBD",  # invert flip within sdpa
            "dUNBD": "pUNBD"  # invert flip within sdpa
        }

        if (reverse_primal_dual):
            sdpainfo['phasevalue'] = REVERSE_PRIMAL_DUAL[sdpainfo['phasevalue']]

        solution = {}
        solution[s.STATUS] = self.STATUS_MAP[sdpainfo['phasevalue']]

        if solution[s.STATUS] in s.SOLUTION_PRESENT:
            x = x.toarray()
            y = y.toarray()
            solution[s.VALUE] = sdpainfo['primalObj']
            solution[s.PRIMAL] = x
            solution[s.EQ_DUAL] = y[:dims['f']]
            solution[s.INEQ_DUAL] = y[dims['f']:]

        return solution
