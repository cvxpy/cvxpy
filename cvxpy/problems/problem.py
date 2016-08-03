"""
Copyright 2013 Steven Diamond

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

import cvxpy.settings as s
import cvxpy.utilities as u
import cvxpy.interface as intf
from cvxpy.error import SolverError, DCPError
from cvxpy.constraints import EqConstraint, LeqConstraint, PSDConstraint
from cvxpy.problems.objective import Minimize, Maximize
from cvxpy.problems.solvers.solver import Solver
from cvxpy.problems.solvers.utilities import SOLVERS
from cvxpy.problems.problem_data.problem_data import ProblemData
# Only need to import cvxpy.transform.get_separable_problems, but this creates
# a circular import (cvxpy.transforms imports Problem). Hence we need to import
# cvxpy here.
import cvxpy

# For representing numpy.matrix objects in hashable form

import multiprocess as multiprocessing
import numpy as np
from collections import namedtuple

# Used in self._cached_data to check if problem's objective or constraints have
# changed.
CachedProblem = namedtuple('CachedProblem', ['objective', 'constraints'])

# Used by pool.map to send solve result back.
SolveResult = namedtuple(
    'SolveResult', ['opt_value', 'status', 'primal_values', 'dual_values'])


class Problem(u.Canonical):
    """A convex optimization problem.

    Attributes
    ----------
    objective : Minimize or Maximize
        The expression to minimize or maximize.
    constraints : list
        The constraints on the problem variables.
    """

    # The solve methods available.
    REGISTERED_SOLVE_METHODS = {}

    def __init__(self, objective, constraints=None):
        if constraints is None:
            constraints = []
        # Check that objective is Minimize or Maximize.
        if not isinstance(objective, (Minimize, Maximize)):
            raise DCPError("Problem objective must be Minimize or Maximize.")
        # Constraints and objective are immutable.
        self.objective = objective
        self.constraints = constraints
        self._value = None
        self._status = None
        # Cached processed data for each solver.
        self._cached_data = {}
        self._reset_cache()
        # List of separable (sub)problems
        self._separable_problems = None
        # Information about the size of the problem and its constituent parts
        self._size_metrics = SizeMetrics(self)

    def _reset_cache(self):
        """Resets the cached data.
        """
        for solver_name in SOLVERS.keys():
            self._cached_data[solver_name] = ProblemData()
        self._cached_data[s.PARALLEL] = CachedProblem(None, None)

    @property
    def value(self):
        """The value from the last time the problem was solved.

        Returns
        -------
        float or None
        """
        return self._value

    @property
    def status(self):
        """The status from the last time the problem was solved.

        Returns
        -------
        str
        """
        return self._status

    def is_dcp(self):
        """Does the problem satisfy DCP rules?
        """
        return all(exp.is_dcp() for exp in self.constraints + [self.objective])

    def canonicalize(self):
        """Computes the graph implementation of the problem.

        Returns
        -------
        tuple
            (affine objective,
             constraints dict)
        """
        canon_constr = []
        obj, constr = self.objective.canonical_form
        canon_constr += constr

        for constr in self.constraints:
            canon_constr += constr.canonical_form[1]

        return (obj, canon_constr)

    def variables(self):
        """Returns a list of the variables in the problem.
        """
        vars_ = self.objective.variables()
        for constr in self.constraints:
            vars_ += constr.variables()
        # Remove duplicates.
        return list(set(vars_))

    def parameters(self):
        """Returns a list of the parameters in the problem.
        """
        params = self.objective.parameters()
        for constr in self.constraints:
            params += constr.parameters()
        # Remove duplicates.
        return list(set(params))

    def constants(self):
        """Returns a list of the constants in the problem.
        """
        const_dict = {}
        constants_ = self.objective.constants()
        for constr in self.constraints:
            constants_ += constr.constants()
        # Remove duplicates.
        # Note that numpy matrices are not hashable, so we use the buildin function id
        const_dict = {id(constant): constant for constant in constants_}
        return list(const_dict.values())

    @property
    def size_metrics(self):
        """Returns an object containing information about the size of the problem.
        """
        return self._size_metrics

    def solve(self, *args, **kwargs):
        """Solves the problem using the specified method.

        Parameters
        ----------
        method : function
            The solve method to use.
        solver : str, optional
            The solver to use.
        verbose : bool, optional
            Overrides the default of hiding solver output.
        solver_specific_opts : dict, optional
            A dict of options that will be passed to the specific solver.
            In general, these options will override any default settings
            imposed by cvxpy.

        Returns
        -------
        float
            The optimal value for the problem, or a string indicating
            why the problem could not be solved.
        """
        func_name = kwargs.pop("method", None)
        if func_name is not None:
            func = Problem.REGISTERED_SOLVE_METHODS[func_name]
            return func(self, *args, **kwargs)
        else:
            return self._solve(*args, **kwargs)

    @classmethod
    def register_solve(cls, name, func):
        """Adds a solve method to the Problem class.

        Parameters
        ----------
        name : str
            The keyword for the method.
        func : function
            The function that executes the solve method.
        """
        cls.REGISTERED_SOLVE_METHODS[name] = func

    def get_problem_data(self, solver):
        """Returns the problem data used in the call to the solver.

        Parameters
        ----------
        solver : str
            The solver the problem data is for.

        Returns
        -------
        tuple
            arguments to solver
        """
        objective, constraints = self.canonicalize()
        # Raise an error if the solver cannot handle the problem.
        SOLVERS[solver].validate_solver(constraints)
        return SOLVERS[solver].get_problem_data(objective, constraints,
                                                self._cached_data)

    def _solve(self,
               solver=None,
               ignore_dcp=False,
               warm_start=False,
               verbose=False,
               parallel=False, **kwargs):
        """Solves a DCP compliant optimization problem.

        Saves the values of primal and dual variables in the variable
        and constraint objects, respectively.

        Parameters
        ----------
        solver : str, optional
            The solver to use. Defaults to ECOS.
        ignore_dcp : bool, optional
            Overrides the default of raising an exception if the problem is not
            DCP.
        warm_start : bool, optional
            Should the previous solver result be used to warm start?
        verbose : bool, optional
            Overrides the default of hiding solver output.
        parallel : bool, optional
            If problem is separable, solve in parallel.
        kwargs : dict, optional
            A dict of options that will be passed to the specific solver.
            In general, these options will override any default settings
            imposed by cvxpy.

        Returns
        -------
        float
            The optimal value for the problem, or a string indicating
            why the problem could not be solved.
        """
        if not self.is_dcp():
            if ignore_dcp:
                print("Problem does not follow DCP rules. "
                      "Solving a convex relaxation.")
            else:
                raise DCPError("Problem does not follow DCP rules.")

        # Problem is linearly constrained least squares
        if solver is None and SOLVERS[s.LS].suitable(self):
            solver = s.LS

        if solver == s.LS:
            solver = SOLVERS[s.LS]
            solver.validate_solver(self)

            objective = self.objective
            constraints = self.constraints

            sym_data = solver.get_sym_data(objective, constraints)
            results_dict = solver.solve(objective, constraints,
                                        self._cached_data, warm_start, verbose,
                                        kwargs)
            self._update_problem_state(results_dict, sym_data, solver)
            return self.value

        # Standard cone problem
        objective, constraints = self.canonicalize()

        # Solve in parallel
        if parallel:
            # Check if the objective or constraint has changed

            if (objective != self._cached_data[s.PARALLEL].objective or
                    constraints != self._cached_data[s.PARALLEL].constraints):
                self._separable_problems = cvxpy.transforms.get_separable_problems(self)
                self._cached_data[s.PARALLEL] = CachedProblem(objective,
                                                              constraints)
            if len(self._separable_problems) > 1:
                return self._parallel_solve(solver, ignore_dcp, warm_start,
                                            verbose, **kwargs)

        # Choose a solver/check the chosen solver.
        if solver is None:
            solver_name = Solver.choose_solver(constraints)
            solver = SOLVERS[solver_name]
        elif solver in SOLVERS:
            solver = SOLVERS[solver]
            solver.validate_solver(constraints)
        else:
            raise SolverError("Unknown solver.")

        sym_data = solver.get_sym_data(objective, constraints,
                                       self._cached_data)
        # Presolve couldn't solve the problem.
        if sym_data.presolve_status is None:
            results_dict = solver.solve(objective, constraints,
                                        self._cached_data, warm_start, verbose,
                                        kwargs)
        # Presolve determined problem was unbounded or infeasible.
        else:
            results_dict = {s.STATUS: sym_data.presolve_status}
        self._update_problem_state(results_dict, sym_data, solver)
        return self.value

    def _parallel_solve(self,
                        solver=None,
                        ignore_dcp=False,
                        warm_start=False,
                        verbose=False, **kwargs):
        """Solves a DCP compliant optimization problem in parallel.

        Saves the values of primal and dual variables in the variable
        and constraint objects, respectively.

        Parameters
        ----------
        solver : str, optional
            The solver to use. Defaults to ECOS.
        ignore_dcp : bool, optional
            Overrides the default of raising an exception if the problem is not
            DCP.
        warm_start : bool, optional
            Should the previous solver result be used to warm start?
        verbose : bool, optional
            Overrides the default of hiding solver output.
        kwargs : dict, optional
            A dict of options that will be passed to the specific solver.
            In general, these options will override any default settings
            imposed by cvxpy.

        Returns
        -------
        float
            The optimal value for the problem, or a string indicating
            why the problem could not be solved.
        """
        def _solve_problem(problem):
            """Solve a problem and then return the optimal value, status,
            primal values, and dual values.
            """
            opt_value = problem.solve(solver=solver,
                                      ignore_dcp=ignore_dcp,
                                      warm_start=warm_start,
                                      verbose=verbose,
                                      parallel=False, **kwargs)
            status = problem.status
            primal_values = [var.value for var in problem.variables()]
            dual_values = [constr.dual_value for constr in problem.constraints]
            return SolveResult(opt_value, status, primal_values, dual_values)

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        solve_results = pool.map(_solve_problem, self._separable_problems)
        pool.close()
        pool.join()
        statuses = {solve_result.status for solve_result in solve_results}
        # Check if at least one subproblem is infeasible or inaccurate
        for status in s.INF_OR_UNB:
            if status in statuses:
                self._handle_no_solution(status)
                break
        else:
            for subproblem, solve_result in zip(self._separable_problems,
                                                solve_results):
                for var, primal_value in zip(subproblem.variables(),
                                             solve_result.primal_values):
                    var.save_value(primal_value)
                for constr, dual_value in zip(subproblem.constraints,
                                              solve_results):
                    constr.save_value(dual_value)
            self._value = sum(solve_result.opt_value
                              for solve_result in solve_results)
            if s.OPTIMAL_INACCURATE in statuses:
                self._status = s.OPTIMAL_INACCURATE
            else:
                self._status = s.OPTIMAL
        return self._value

    def _update_problem_state(self, results_dict, sym_data, solver):
        """Updates the problem state given the solver results.

        Updates problem.status, problem.value and value of
        primal and dual variables.

        Parameters
        ----------
        results_dict : dict
            A dictionary containing the solver results.
        sym_data : SymData
            The symbolic data for the problem.
        solver : Solver
            The solver type used to obtain the results.
        """
        if results_dict[s.STATUS] in s.SOLUTION_PRESENT:
            self._save_values(results_dict[s.PRIMAL], self.variables(),
                              sym_data.var_offsets)
            # Not all solvers provide dual variables.
            if s.EQ_DUAL in results_dict:
                self._save_dual_values(results_dict[s.EQ_DUAL],
                                       sym_data.constr_map[s.EQ],
                                       [EqConstraint])
            if s.INEQ_DUAL in results_dict:
                self._save_dual_values(results_dict[s.INEQ_DUAL],
                                       sym_data.constr_map[s.LEQ],
                                       [LeqConstraint, PSDConstraint])
            # Correct optimal value if the objective was Maximize.
            value = results_dict[s.VALUE]
            self._value = self.objective.primal_to_result(value)
        # Infeasible or unbounded.
        elif results_dict[s.STATUS] in s.INF_OR_UNB:
            self._handle_no_solution(results_dict[s.STATUS])
        # Solver failed to solve.
        else:
            raise SolverError(
                "Solver '%s' failed. Try another solver." % solver.name())
        self._status = results_dict[s.STATUS]

    def unpack_results(self, solver_name, results_dict):
        """Parses the output from a solver and updates the problem state.

        Updates problem.status, problem.value and value of
        primal and dual variables.

        Assumes the results are from the given solver.

        Parameters
        ----------
        solver_name : str
            The name of the solver being used.
        results_dict : dict
            The solver output.
        """
        if solver_name not in SOLVERS:
            raise SolverError("Unknown solver.")
        solver = SOLVERS[solver_name]

        objective, constraints = self.canonicalize()
        sym_data = solver.get_sym_data(objective, constraints,
                                       self._cached_data)
        data = {s.DIMS: sym_data.dims, s.OFFSET: 0}
        results_dict = solver.format_results(results_dict, data,
                                             self._cached_data)
        self._update_problem_state(results_dict, sym_data, solver)

    def _handle_no_solution(self, status):
        """Updates value fields when the problem is infeasible or unbounded.

        Parameters
        ----------
        status: str
            The status of the solver.
        """
        # Set all primal and dual variable values to None.
        for var_ in self.variables():
            var_.save_value(None)
        for constr in self.constraints:
            constr.save_value(None)
        # Set the problem value.
        if status in [s.INFEASIBLE, s.INFEASIBLE_INACCURATE]:
            self._value = self.objective.primal_to_result(np.inf)
        elif status in [s.UNBOUNDED, s.UNBOUNDED_INACCURATE]:
            self._value = self.objective.primal_to_result(-np.inf)

    def _save_dual_values(self, result_vec, constraints, constr_types):
        """Saves the values of the dual variables.

        Parameters
        ----------
        result_vec : array_like
            A vector containing the dual variable values.
        constraints : list
            A list of the LinEqConstr/LinLeqConstr in the problem.
        constr_types : type
            A list of constraint types to consider.
        """
        constr_offsets = {}
        offset = 0
        for constr in constraints:
            constr_offsets[constr.constr_id] = offset
            offset += constr.size[0] * constr.size[1]
        active_constraints = []
        for constr in self.constraints:
            # Ignore constraints of the wrong type.
            if type(constr) in constr_types:
                active_constraints.append(constr)
        self._save_values(result_vec, active_constraints, constr_offsets)

    def _save_values(self, result_vec, objects, offset_map):
        """Saves the values of the optimal primal/dual variables.

        Parameters
        ----------
        results_vec : array_like
            A vector containing the variable values.
        objects : list
            The variables or constraints where the values will be stored.
        offset_map : dict
            A map of object id to offset in the results vector.
        """
        if len(result_vec) > 0:
            # Cast to desired matrix type.
            result_vec = intf.DEFAULT_INTF.const_to_matrix(result_vec)
        for obj in objects:
            rows, cols = obj.size
            if obj.id in offset_map:
                offset = offset_map[obj.id]
                # Handle scalars
                if (rows, cols) == (1, 1):
                    value = intf.index(result_vec, (offset, 0))
                else:
                    value = intf.DEFAULT_INTF.zeros(rows, cols)
                    intf.DEFAULT_INTF.block_add(
                        value, result_vec[offset:offset + rows * cols], 0, 0,
                        rows, cols)
                offset += rows * cols
            else:  # The variable was multiplied by zero.
                value = intf.DEFAULT_INTF.zeros(rows, cols)
            obj.save_value(value)

    def __str__(self):
        if len(self.constraints) == 0:
            return str(self.objective)
        else:
            subject_to = "subject to "
            lines = [str(self.objective),
                     subject_to + str(self.constraints[0])]
            for constr in self.constraints[1:]:
                lines += [len(subject_to) * " " + str(constr)]
            return '\n'.join(lines)

    def __repr__(self):
        return "Problem(%s, %s)" % (repr(self.objective),
                                    repr(self.constraints))

    def __neg__(self):
        return Problem(-self.objective, self.constraints)

    def __add__(self, other):
        if other == 0:
            return self
        elif not isinstance(other, Problem):
            return NotImplemented
        return Problem(self.objective + other.objective,
                       list(set(self.constraints + other.constraints)))

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return NotImplemented

    def __sub__(self, other):
        if not isinstance(other, Problem):
            return NotImplemented
        return Problem(self.objective - other.objective,
                       list(set(self.constraints + other.constraints)))

    def __rsub__(self, other):
        if other == 0:
            return -self
        else:
            return NotImplemented

    def __mul__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        return Problem(self.objective * other, self.constraints)

    __rmul__ = __mul__

    def __div__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        return Problem(self.objective * (1.0 / other), self.constraints)

    __truediv__ = __div__


class SizeMetrics(object):
    """Reports various metrics regarding the problem

    Attributes
    ----------

    Counts:
        num_scalar_variables:
            The number of scalar variables in the problem.
        num_scalar_data
            The number of scalar constants and parameters in the problem. The number of
            constants used across all matrices, vectors, in the problem.
            Some constants are not apparent when the problem is constructed: for example,
            The sum_squares expression is a wrapper for a quad_over_lin expression with a
            constant 1 in the denominator.
        num_scalar_eq_constr:
            The number of scalar equality constraints in the problem.
        num_scalar_leq_constr:
            The number of scalar inequality constraints in the problem.

    Max and min sizes:
        max_data_dimension:
            The longest dimension of any data block constraint or parameter.

    """

    def __init__(self, problem):
        # num_scalar_variables
        self.num_scalar_variables = 0
        for var in problem.variables():
            self.num_scalar_variables += np.prod(var.size)

        # num_scalar_data and max_data_dimension
        self.max_data_dimension = 0
        self.num_scalar_data = 0
        for const in problem.constants()+problem.parameters():
            thismax = 0
            # Compute number of data
            self.num_scalar_data += np.prod(const.size)
            thismax = max(const.size)

            # Get max absolute residual:
            if self.max_data_dimension < thismax:
                self.max_data_dimension = thismax

        # num_scalar_eq_constr
        self.num_scalar_eq_constr = 0
        for constraint in problem.constraints:
            if constraint.__class__.__name__ is "EqConstraint":
                self.num_scalar_eq_constr += np.prod(constraint._expr.size)

        # num_scalar_leq_constr
        self.num_scalar_leq_constr = 0
        for constraint in problem.constraints:
            if constraint.__class__.__name__ is "LeqConstraint":
                self.num_scalar_leq_constr += np.prod(constraint._expr.size)
