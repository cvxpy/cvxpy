"""
Copyright 2013 Steven Diamond, 2017 Akshay Agrawal

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
from cvxpy.error import DCPError, SolverError
# from cvxpy.expressions.variables import Bool, Int
from cvxpy.problems.objective import Minimize, Maximize
from cvxpy.reductions.solvers.solving_chain import construct_solving_chain
from cvxpy.interface.matrix_utilities import scalar_value

# TODO(akshayka): This is a hack. Fix this if possible.
# Only need to import cvxpy.transform.get_separable_problems, but this creates
# a circular import (cvxpy.transforms imports Problem). Hence we need to import
# cvxpy here.
import cvxpy  # noqa
import cvxpy.constraints.zero as eqc
import cvxpy.utilities as u
from collections import namedtuple
import multiprocess as multiprocessing


SolveResult = namedtuple(
    'SolveResult',
    ['opt_value', 'status', 'primal_values', 'dual_values'])


class Problem(u.Canonical):
    """A convex optimization problem.

    Problems are immutable, save for modification through the specification
    of :class:`~cvxpy.expressions.constants.parameters.Parameter`

    Parameters
    ----------
    objective : Minimize or Maximize
        The problem's objective.
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
        self._objective = objective
        self._constraints = [c for c in constraints]
        # Cache the variables as a list.
        self._vars = self._variables()
        self._value = None
        self._status = None
        # The solving chain with which to solve the problem
        self._solving_chain = None
        # List of separable (sub)problems
        self._separable_problems = None
        # Information about the shape of the problem and its constituent parts
        self._size_metrics = SizeMetrics(self)
        # Benchmarks reported by the solver:
        self._solver_stats = None
        self.args = [self._objective, self._constraints]
        # Cache for warm start.
        self._solver_cache = {}

    @property
    def value(self):
        """float : The value from the last time the problem was solved
                   (or None if not solved).
        """
        if self._value is None:
            return None
        else:
            return scalar_value(self._value)

    @property
    def status(self):
        """str : The status from the last time the problem was solved; one
                 of optimal, infeasible, or unbounded.
        """
        return self._status

    @property
    def objective(self):
        """Minimize or Maximize : The problem's objective.

        Note that the objective cannot be reassigned after creation,
        and modifying the objective after creation will result in
        undefined behavior.
        """
        return self._objective

    @property
    def constraints(self):
        """A shallow copy of the problem's constraints.

        Note that constraints cannot be reassigned, appended to, or otherwise
        modified after creation, except through parameters.
        """
        return self._constraints[:]

    def is_dcp(self):
        """Does the problem satisfy DCP rules?
        """
        return all(exp.is_dcp() for exp in self.constraints + [self.objective])

    def is_qp(self):
        """Is problem a quadratic program?
        """
        for c in self.constraints:
            if not (isinstance(c, eqc.Zero) or c.args[0].is_pwl()):
                return False
        for var in self.variables():
            if var.is_psd() or var.is_nsd():
                return False
        return (self.is_dcp() and self.objective.args[0].is_qpwa())

    def is_mixed_integer(self):
        return any(v.attributes['boolean'] or v.attributes['integer']
                   for v in self.variables())

    def variables(self):
        """Accessor method for variables.

        Returns
        -------
        list of :class:`~cvxpy.expressions.variable.Variable`
            A list of the variables in the problem.
        """
        return self._vars

    def _variables(self):
        vars_ = self.objective.variables()
        for constr in self.constraints:
            vars_ += constr.variables()
        seen = set()
        # never use list as a variable name
        return [seen.add(obj.id) or obj for obj in vars_ if obj.id not in seen]

    def parameters(self):
        """Accessor method for parameters.

        Returns
        -------
        list of :class:`~cvxpy.expressions.constants.parameter.Parameter`
            A list of the parameters in the problem.
        """
        params = self.objective.parameters()
        for constr in self.constraints:
            params += constr.parameters()
        return list(set(params))

    def constants(self):
        """Accessor method for parameters.

        Returns
        -------
        list of :class:`~cvxpy.expressions.constants.constant.Constant`
            A list of the constants in the problem.
        """
        const_dict = {}
        constants_ = self.objective.constants()
        for constr in self.constraints:
            constants_ += constr.constants()
        # Note that numpy matrices are not hashable, so we use the built-in
        # function "id"
        const_dict = {id(constant): constant for constant in constants_}
        return list(const_dict.values())

    def atoms(self):
        """Accessor method for atoms.

        Returns
        -------
        list of :class:`~cvxpy.atoms.Atom`
            A list of the atom types in the problem; note that this list
            contains classes, not instances.
        """
        atoms = self.objective.atoms()
        for constr in self.constraints:
            atoms += constr.atoms()
        return list(set(atoms))

    @property
    def size_metrics(self):
        """:class:`~cvxpy.problems.problem.SizeMetrics` : Information about the problem's size.
        """
        return self._size_metrics

    @property
    def solver_stats(self):
        """:class:`~cvxpy.problems.problem.SolverStats` : Information returned by the solver.
        """
        return self._solver_stats

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

        Raises
        ------
        DCPError
            Raised if the problem is not DCP.
        SolverError
            Raised if no suitable solver exists among the installed solvers,
            or if an unanticipated error is encountered.
        """
        func_name = kwargs.pop("method", None)
        if func_name is not None:
            solve_func = Problem.REGISTERED_SOLVE_METHODS[func_name]
        else:
            solve_func = Problem._solve
        return solve_func(self, *args, **kwargs)

    @classmethod
    def register_solve(cls, name, func):
        """Adds a solve method to the Problem class.

        Parameters
        ----------
        name : str
            The keyword for the method.
        func : function
            The function that executes the solve method. This function must
            take as its first argument the problem instance to solve.
        """
        cls.REGISTERED_SOLVE_METHODS[name] = func

    def get_problem_data(self, solver):
        """Returns the problem data used in the call to the solver.

        When a problem is solved, a chain of reductions, called a
        :class:`~cvxpy.reductions.solvers.solving_chain.SolvingChain`,
        compiles it to some low-level representation that is compatible with the
        targeted solver. This method returns that low-level representation.

        For some solving chains, this low-level representation is a dictionary
        that contains exactly those arguments that were supplied to the solver;
        however, for other solving chains, the data is an intermediate
        representation that is compiled even further by libraries other than
        CVXPY.

        A solution to the equivalent low-level problem can be obtained via the
        data by invoking the solve_via_data method of the returned solving
        chain, a thin wrapper around the code external to CVXPY that further
        processes and solves the problem. Invoke the unpack_results method
        to recover a solution to the original problem.

        Parameters
        ----------
        solver : str
            The solver the problem data is for.

        Returns
        -------
        dict or object
            lowest level representation of problem
        SolvingChain
            The solving chain that created the data.
        list
            The inverse data generated by the chain.
        """
        try:
            solving_chain = construct_solving_chain(self, solver)
        except Exception as e:
            raise e
        data, inv_data = solving_chain.apply(self)
        return data, solving_chain, inv_data

    def _solve(self,
               solver=None,
               ignore_dcp=False,
               warm_start=True,
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
            TODO(akshayka): This option will probably be eliminated.
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
        if parallel:
            from cvxpy.transforms.separable_problems import get_separable_problems
            self._separable_problems = (get_separable_problems(self))
            if len(self._separable_problems) > 1:
                return self._parallel_solve(solver, ignore_dcp, warm_start,
                                            verbose, **kwargs)

        # If a previous chain does not exist, or if the solver is
        # specified and it does not match the solver used for the previous
        # solve, then construct a new solving chain.
        if (self._solving_chain is None
                or (solver is not None
                    and self._solving_chain.solver.name != solver)):
            try:
                self._solving_chain = construct_solving_chain(self,
                                                              solver=solver)
            except Exception as e:
                raise e

        data, inverse_data = self._solving_chain.apply(self)
        solution = self._solving_chain.solve_via_data(self, data, warm_start, verbose,
                                                      kwargs)
        self.unpack_results(solution, self._solving_chain, inverse_data)
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

    def unpack_results(self, solution, chain, inverse_data):
        """Updates the problem state given the solver results.

        Updates problem.status, problem.value and value of
        primal and dual variables.

        Parameters
        __________
        solution : Solution
            The solution returned by applying the chain to the problem
            and invoking the solver on the resulting data.
        chain : SolvingChain
            A solving chain that was used to solve the problem.
        inverse_data : list
            The inverse data returned by applying the chain to the problem.
        """
        solution = chain.invert(solution, inverse_data)
        self._value = solution.opt_val
        if solution.status in s.SOLUTION_PRESENT:
            for v in self.variables():
                v.save_value(solution.primal_vars[v.id])
            for c in self.constraints:
                if c.id in solution.dual_vars:
                    c.save_value(solution.dual_vars[c.id])
        elif solution.status in s.INF_OR_UNB:
            for v in self.variables():
                v.save_value(None)
            for constr in self.constraints:
                constr.save_value(None)
        else:
            raise SolverError(
                "Solver '%s' failed. " % chain.solver.name() +
                "Try another solver or solve with verbose=True for more information. " +
                "Try recentering the problem data around 0 and rescaling " +
                "to reduce the dynamic range."
            )
        self._status = solution.status
        self._solver_stats = SolverStats(solution.attr, chain.solver.name())

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


# TODO(akshayka): Consider moving this to another file
class SolverStats(object):
    """Reports some of the miscellaneous information that is returned
    by the solver after solving but that is not captured directly by
    the Problem instance.

    Attributes
    ----------
    solve_time : double
        The time (in seconds) it took for the solver to solve the problem.
    setup_time : double
        The time (in seconds) it took for the solver to setup the problem.
    num_iters : int
        The number of iterations the solver had to go through to find a solution.
    """
    def __init__(self, results_dict, solver_name):
        self.solver_name = solver_name
        self.solve_time = None
        self.setup_time = None
        self.num_iters = None

        if s.SOLVE_TIME in results_dict:
            self.solve_time = results_dict[s.SOLVE_TIME]
        if s.SETUP_TIME in results_dict:
            self.setup_time = results_dict[s.SETUP_TIME]
        if s.NUM_ITERS in results_dict:
            self.num_iters = results_dict[s.NUM_ITERS]


# TODO(akshayka): Consider moving this to another file
class SizeMetrics(object):
    """Reports various metrics regarding the problem.

    Attributes
    ----------

    num_scalar_variables : integer
        The number of scalar variables in the problem.
    num_scalar_data : integer
        The number of scalar constants and parameters in the problem. The number of
        constants used across all matrices, vectors, in the problem.
        Some constants are not apparent when the problem is constructed: for example,
        The sum_squares expression is a wrapper for a quad_over_lin expression with a
        constant 1 in the denominator.
    num_scalar_eq_constr : integer
        The number of scalar equality constraints in the problem.
    num_scalar_leq_constr : integer
        The number of scalar inequality constraints in the problem.

    max_data_dimension : integer
        The longest dimension of any data block constraint or parameter.
    max_big_small_squared : integer
        The maximum value of (big)(small)^2 over all data blocks of the problem, where
        (big) is the larger dimension and (small) is the smaller dimension
        for each data block.
    """

    def __init__(self, problem):
        # num_scalar_variables
        self.num_scalar_variables = 0
        for var in problem.variables():
            self.num_scalar_variables += var.size

        # num_scalar_data, max_data_dimension, and max_big_small_squared
        self.max_data_dimension = 0
        self.num_scalar_data = 0
        self.max_big_small_squared = 0
        for const in problem.constants()+problem.parameters():
            big = 0
            # Compute number of data
            self.num_scalar_data += const.size
            big = 1 if len(const.shape) == 0 else max(const.shape)
            small = 1 if len(const.shape) == 0 else min(const.shape)

            # Get max data dimension:
            if self.max_data_dimension < big:
                self.max_data_dimension = big

            if self.max_big_small_squared < big*small**2:
                self.max_big_small_squared = big*small**2

        # num_scalar_eq_constr
        self.num_scalar_eq_constr = 0
        for constraint in problem.constraints:
            if constraint.__class__.__name__ is "Zero":
                self.num_scalar_eq_constr += constraint.args[0].size

        # num_scalar_leq_constr
        self.num_scalar_leq_constr = 0
        for constraint in problem.constraints:
            if constraint.__class__.__name__ is "NonPos":
                self.num_scalar_leq_constr += constraint.args[0].size
