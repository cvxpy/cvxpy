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

from cvxpy import settings as s, Constant
from cvxpy import error
from cvxpy.expressions import cvxtypes
from cvxpy.problems.objective import Minimize, Maximize
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.dgp2dcp.dgp2dcp import Dgp2Dcp
from cvxpy.reductions.dqcp2dcp import dqcp2dcp
from cvxpy.reductions.eval_params import EvalParams
from cvxpy.reductions.flip_objective import FlipObjective
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
from cvxpy.reductions.solvers.defines import SOLVER_MAP_QP, SOLVER_MAP_CONIC
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.reductions.solvers.solving_chain import construct_solving_chain
from cvxpy.interface.matrix_utilities import scalar_value
from cvxpy.reductions.solvers import bisection
from cvxpy.reductions.solvers import defines as slv_def
from cvxpy.settings import SOLVERS
from cvxpy.utilities.deterministic import unique_list
import cvxpy.utilities.performance_utils as perf
from cvxpy.constraints import Equality, Inequality, NonPos, Zero, NonNeg
import cvxpy.utilities as u

from collections import namedtuple
import numpy as np
import time
import warnings


SolveResult = namedtuple(
    'SolveResult',
    ['opt_value', 'status', 'primal_values', 'dual_values'])


_COL_WIDTH = 79
_HEADER = (
    '='*_COL_WIDTH +
    '\n' +
    ('CVXPY').center(_COL_WIDTH) +
    '\n' +
    ('v' + cvxtypes.version()).center(_COL_WIDTH) +
    '\n' +
    '='*_COL_WIDTH
)
_COMPILATION_STR = (
    '-'*_COL_WIDTH +
    '\n' +
    ('Compilation').center(_COL_WIDTH) +
    '\n' +
    '-'*_COL_WIDTH
)
_NUM_SOLVER_STR = (
    '-'*_COL_WIDTH +
    '\n' +
    ('Numerical solver').center(_COL_WIDTH) +
    '\n' +
    '-'*_COL_WIDTH
)
_FOOTER = (
    '-'*_COL_WIDTH +
    '\n' +
    ('Summary').center(_COL_WIDTH) +
    '\n' +
    '-'*_COL_WIDTH
)


class Cache:
    def __init__(self) -> None:
        self.key = None
        self.solving_chain = None
        self.param_prog = None
        self.inverse_data = None

    def invalidate(self) -> None:
        self.key = None
        self.solving_chain = None
        self.param_prog = None
        self.inverse_data = None

    def make_key(self, solver, gp):
        return (solver, gp)

    def gp(self):
        return self.key is not None and self.key[1]


class Problem(u.Canonical):
    """A convex optimization problem.

    Problems are immutable, save for modification through the specification
    of :class:`~cvxpy.expressions.constants.parameters.Parameter`

    Arguments
    ---------
    objective : Minimize or Maximize
        The problem's objective.
    constraints : list
        The constraints on the problem variables.
    """

    # The solve methods available.
    REGISTERED_SOLVE_METHODS = {}

    def __init__(self, objective, constraints=None) -> None:
        if constraints is None:
            constraints = []
        # Check that objective is Minimize or Maximize.
        if not isinstance(objective, (Minimize, Maximize)):
            raise error.DCPError("Problem objective must be Minimize or Maximize.")
        # Constraints and objective are immutable.
        self._objective = objective

        def bool_value_filter(cstr_expr):
            if not isinstance(cstr_expr, bool):
                return cstr_expr
            # replace `True` or `False` values with equivalent Expressions.
            return Constant(0) <= Constant(1) if cstr_expr else Constant(1) <= Constant(0)

        self._constraints = list(map(bool_value_filter, constraints))
        self._value = None
        self._status = None
        self._solution = None
        self._cache = Cache()
        self._solver_cache = {}
        # Information about the shape of the problem and its constituent parts
        self._size_metrics = None
        # Benchmarks reported by the solver:
        self._solver_stats = None
        self._compilation_time = None
        self._solve_time = None
        self.args = [self._objective, self._constraints]

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
                 of optimal, infeasible, or unbounded (with or without
                 suffix inaccurate).
        """
        return self._status

    @property
    def solution(self):
        """Solution : The solution from the last time the problem was solved.
        """
        return self._solution

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

    @property
    def param_dict(self):
        """
        Expose all parameters as a dictionary
        """
        return {parameters.name(): parameters for parameters in self.parameters()}

    @property
    def var_dict(self):
        """
        Expose all variables as a dictionary
        """
        return {variable.name(): variable for variable in self.variables()}

    @perf.compute_once
    def is_dcp(self, dpp: bool = False) -> bool:
        """Does the problem satisfy DCP rules?

        Arguments
        ---------
        dpp : bool, optional
            If True, enforce the disciplined parametrized programming (DPP)
            ruleset; only relevant when the problem involves Parameters.
            DPP is a mild restriction of DCP. When a problem involving
            Parameters is DPP, subsequent solves can be much faster than
            the first one. For more information, consult the documentation at

            https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming

        Returns
        -------
        bool
            True if the Expression is DCP, False otherwise.
        """
        return all(
          expr.is_dcp(dpp) for expr in self.constraints + [self.objective])

    @perf.compute_once
    def is_dgp(self, dpp: bool = False) -> bool:
        """Does the problem satisfy DGP rules?

        Arguments
        ---------
        dpp : bool, optional
            If True, enforce the disciplined parametrized programming (DPP)
            ruleset; only relevant when the problem involves Parameters.
            DPP is a mild restriction of DGP. When a problem involving
            Parameters is DPP, subsequent solves can be much faster than
            the first one. For more information, consult the documentation at

            https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming

        Returns
        -------
        bool
            True if the Expression is DGP, False otherwise.
        """
        return all(
          expr.is_dgp(dpp) for expr in self.constraints + [self.objective])

    @perf.compute_once
    def is_dqcp(self) -> bool:
        """Does the problem satisfy the DQCP rules?
        """
        return all(
          expr.is_dqcp() for expr in self.constraints + [self.objective])

    @perf.compute_once
    def is_dpp(self, context='dcp') -> bool:
        """Does the problem satisfy DPP rules?

        DPP is a mild restriction of DGP. When a problem involving
        Parameters is DPP, subsequent solves can be much faster than
        the first one. For more information, consult the documentation at

        https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming

        Arguments
        ---------
        context : str
            Whether to check DPP-compliance for DCP or DGP; ``context`` should
            be either ``'dcp'`` or ``'dgp'``. Calling ``problem.is_dpp('dcp')``
            is equivalent to ``problem.is_dcp(dpp=True)``, and
            `problem.is_dpp('dgp')`` is equivalent to
            `problem.is_dgp(dpp=True)`.

        Returns
        -------
        bool
            Whether the problem satisfies the DPP rules.
        """
        if context.lower() == 'dcp':
            return self.is_dcp(dpp=True)
        elif context.lower() == 'dgp':
            return self.is_dgp(dpp=True)
        else:
            raise ValueError("Unsupported context ", context)

    @perf.compute_once
    def is_qp(self) -> bool:
        """Is problem a quadratic program?
        """
        for c in self.constraints:
            if not (isinstance(c, (Equality, Zero)) or c.args[0].is_pwl()):
                return False
        for var in self.variables():
            if var.is_psd() or var.is_nsd():
                return False
        return (self.is_dcp() and self.objective.args[0].is_qpwa())

    @perf.compute_once
    def is_mixed_integer(self) -> bool:
        return any(v.attributes['boolean'] or v.attributes['integer']
                   for v in self.variables())

    @perf.compute_once
    def variables(self):
        """Accessor method for variables.

        Returns
        -------
        list of :class:`~cvxpy.expressions.variable.Variable`
            A list of the variables in the problem.
        """
        vars_ = self.objective.variables()
        for constr in self.constraints:
            vars_ += constr.variables()
        return unique_list(vars_)

    @perf.compute_once
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
        return unique_list(params)

    @perf.compute_once
    def constants(self):
        """Accessor method for constants.

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
        return unique_list(atoms)

    @property
    def size_metrics(self):
        """:class:`~cvxpy.problems.problem.SizeMetrics` : Information about the problem's size.
        """
        if self._size_metrics is None:
            self._size_metrics = SizeMetrics(self)
        return self._size_metrics

    @property
    def solver_stats(self):
        """:class:`~cvxpy.problems.problem.SolverStats` : Information returned by the solver.
        """
        return self._solver_stats

    def solve(self, *args, **kwargs):
        """Compiles and solves the problem using the specified method.

        Populates the :code:`status` and :code:`value` attributes on the
        problem object as a side-effect.

        Arguments
        ---------
        solver : str, optional
            The solver to use. For example, 'ECOS', 'SCS', or 'OSQP'.
        verbose : bool, optional
            Overrides the default of hiding solver output, and prints
            logging information describing CVXPY's compilation process.
        gp : bool, optional
            If True, parses the problem as a disciplined geometric program
            instead of a disciplined convex program.
        qcp : bool, optional
            If True, parses the problem as a disciplined quasiconvex program
            instead of a disciplined convex program.
        requires_grad : bool, optional
            Makes it possible to compute gradients of a solution with respect to
            Parameters by calling ``problem.backward()`` after solving, or to
            compute perturbations to the variables given perturbations to Parameters by
            calling ``problem.derivative()``.

            Gradients are only supported for DCP and DGP problems, not
            quasiconvex problems. When computing gradients (i.e., when
            this argument is True), the problem must satisfy the DPP rules.
        enforce_dpp : bool, optional
            When True, a DPPError will be thrown when trying to solve a non-DPP
            problem (instead of just a warning). Only relevant for problems
            involving Parameters. Defaults to False.
        method : function, optional
            A custom solve method to use.
        kwargs : keywords, optional
            Additional solver specific arguments. See Notes below.

        Notes
        ------
        CVXPY interfaces with a wide range of solvers; the algorithms used by these solvers
        have arguments relating to stopping criteria, and strategies to improve solution quality.

        There is no one choice of arguments which is perfect for every problem. If you are not
        getting satisfactory results from a solver, you can try changing its arguments. The
        exact way this is done depends on the specific solver. Here are some examples:

            prob.solve(solver='ECOS', abstol=1e-6)
            prob.solve(solver='OSQP', max_iter=10000).
            mydict = {"MSK_DPAR_INTPNT_CO_TOL_NEAR_REL":  10}
            prob.solve(solver='MOSEK', mosek_params=mydict).

        You should refer to CVXPY's web documentation for details on how to pass solver
        solver arguments, available at

        https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options

        Returns
        -------
        float
            The optimal value for the problem, or a string indicating
            why the problem could not be solved.

        Raises
        ------
        cvxpy.error.DCPError
            Raised if the problem is not DCP and `gp` is False.
        cvxpy.error.DGPError
            Raised if the problem is not DGP and `gp` is True.
        cvxpy.error.SolverError
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
    def register_solve(cls, name, func) -> None:
        """Adds a solve method to the Problem class.

        Arguments
        ---------
        name : str
            The keyword for the method.
        func : function
            The function that executes the solve method. This function must
            take as its first argument the problem instance to solve.
        """
        cls.REGISTERED_SOLVE_METHODS[name] = func

    def get_problem_data(self, solver, gp: bool = False, enforce_dpp: bool = False,
                         verbose: bool = False):
        """Returns the problem data used in the call to the solver.

        When a problem is solved, CVXPY creates a chain of reductions enclosed
        in a :class:`~cvxpy.reductions.solvers.solving_chain.SolvingChain`,
        and compiles it to some low-level representation that is
        compatible with the targeted solver. This method returns that low-level
        representation.

        For some solving chains, this low-level representation is a dictionary
        that contains exactly those arguments that were supplied to the solver;
        however, for other solving chains, the data is an intermediate
        representation that is compiled even further by the solver interfaces.

        A solution to the equivalent low-level problem can be obtained via the
        data by invoking the `solve_via_data` method of the returned solving
        chain, a thin wrapper around the code external to CVXPY that further
        processes and solves the problem. Invoke the unpack_results method
        to recover a solution to the original problem.

        For example:

        ::

            objective = ...
            constraints = ...
            problem = cp.Problem(objective, constraints)
            data, chain, inverse_data = problem.get_problem_data(cp.SCS)
            # calls SCS using `data`
            soln = chain.solve_via_data(problem, data)
            # unpacks the solution returned by SCS into `problem`
            problem.unpack_results(soln, chain, inverse_data)

        Alternatively, the `data` dictionary returned by this method
        contains enough information to bypass CVXPY and call the solver
        directly.

        For example:

        ::

            problem = cp.Problem(objective, constraints)
            data, _, _ = problem.get_problem_data(cp.SCS)

            import scs
            probdata = {
              'A': data['A'],
              'b': data['b'],
              'c': data['c'],
            }
            cone_dims = data['dims']
            cones = {
                "f": cone_dims.zero,
                "l": cone_dims.nonpos,
                "q": cone_dims.soc,
                "ep": cone_dims.exp,
                "s": cone_dims.psd,
            }
            soln = scs.solve(data, cones)

        The structure of the data dict that CVXPY returns depends on the
        solver. For details, consult the solver interfaces in
        `cvxpy/reductions/solvers`.

        Arguments
        ---------
        solver : str
            The solver the problem data is for.
        gp : bool, optional
            If True, then parses the problem as a disciplined geometric program
            instead of a disciplined convex program.
        enforce_dpp : bool, optional
            When True, a DPPError will be thrown when trying to parse a non-DPP
            problem (instead of just a warning). Defaults to False.
        verbose : bool, optional
            If True, print verbose output related to problem compilation.

        Returns
        -------
        dict or object
            lowest level representation of problem
        SolvingChain
            The solving chain that created the data.
        list
            The inverse data generated by the chain.
        """
        start = time.time()
        key = self._cache.make_key(solver, gp)
        if key != self._cache.key:
            self._cache.invalidate()
            solving_chain = self._construct_chain(
                solver=solver, gp=gp, enforce_dpp=enforce_dpp)
            self._cache.key = key
            self._cache.solving_chain = solving_chain
            self._solver_cache = {}
        else:
            solving_chain = self._cache.solving_chain

        if verbose:
            print(_COMPILATION_STR)

        if self._cache.param_prog is not None:
            # fast path, bypasses application of reductions
            if verbose:
                s.LOGGER.info(
                        'Using cached ASA map, for faster compilation '
                        '(bypassing reduction chain).')
            if gp:
                dgp2dcp = self._cache.solving_chain.get(Dgp2Dcp)
                # Parameters in the param cone prog are the logs
                # of parameters in the original problem (with one exception:
                # parameters appearing as exponents (in power and gmatmul
                # atoms) are unchanged.
                old_params_to_new_params = dgp2dcp.canon_methods._parameters
                for param in self.parameters():

                    if param in old_params_to_new_params:
                        old_params_to_new_params[param].value = np.log(
                            param.value)

            data, solver_inverse_data = solving_chain.solver.apply(
                self._cache.param_prog)
            inverse_data = self._cache.inverse_data + [solver_inverse_data]
            self._compilation_time = time.time() - start
            if verbose:
                s.LOGGER.info(
                        'Finished problem compilation '
                        '(took %.3e seconds).', self._compilation_time)
        else:
            if verbose:
                solver_name = solving_chain.reductions[-1].name()
                reduction_chain_str = ' -> '.join(
                        type(r).__name__ for r in solving_chain.reductions)
                s.LOGGER.info(
                         'Compiling problem (target solver=%s).', solver_name)
                s.LOGGER.info('Reduction chain: %s', reduction_chain_str)
            data, inverse_data = solving_chain.apply(self, verbose)
            safe_to_cache = (
                isinstance(data, dict)
                and s.PARAM_PROB in data
                and not any(isinstance(reduction, EvalParams)
                            for reduction in solving_chain.reductions)
            )
            self._compilation_time = time.time() - start
            if verbose:
                s.LOGGER.info(
                        'Finished problem compilation '
                        '(took %.3e seconds).', self._compilation_time)
            if safe_to_cache:
                if verbose and self.parameters():
                    s.LOGGER.info(
                        '(Subsequent compilations of this problem, using the '
                        'same arguments, should ' 'take less time.)')
                self._cache.param_prog = data[s.PARAM_PROB]
                # the last datum in inverse_data corresponds to the solver,
                # so we shouldn't cache it
                self._cache.inverse_data = inverse_data[:-1]
        return data, solving_chain, inverse_data

    def _find_candidate_solvers(self,
                                solver=None,
                                gp: bool = False):
        """
        Find candidate solvers for the current problem. If solver
        is not None, it checks if the specified solver is compatible
        with the problem passed.

        Arguments
        ---------
        solver : Union[string, Solver, None]
            The name of the solver with which to solve the problem or an
            instance of a custom solver. If no solver is supplied
            (i.e., if solver is None), then the targeted solver may be any
            of those that are installed. If the problem is variable-free,
            then this parameter is ignored.
        gp : bool
            If True, the problem is parsed as a Disciplined Geometric Program
            instead of as a Disciplined Convex Program.

        Returns
        -------
        dict
            A dictionary of compatible solvers divided in `qp_solvers`
            and `conic_solvers`.

        Raises
        ------
        cvxpy.error.SolverError
            Raised if the problem is not DCP and `gp` is False.
        cvxpy.error.DGPError
            Raised if the problem is not DGP and `gp` is True.
        """
        candidates = {'qp_solvers': [],
                      'conic_solvers': []}
        if isinstance(solver, Solver):
            return self._add_custom_solver_candidates(solver)

        if solver is not None:
            if solver not in slv_def.INSTALLED_SOLVERS:
                raise error.SolverError("The solver %s is not installed." % solver)
            if solver in slv_def.CONIC_SOLVERS:
                candidates['conic_solvers'] += [solver]
            if solver in slv_def.QP_SOLVERS:
                candidates['qp_solvers'] += [solver]
        else:
            candidates['qp_solvers'] = [s for s in slv_def.INSTALLED_SOLVERS
                                        if s in slv_def.QP_SOLVERS]
            candidates['conic_solvers'] = []
            # ECOS_BB can only be called explicitly.
            for slv in slv_def.INSTALLED_SOLVERS:
                if slv in slv_def.CONIC_SOLVERS and slv != s.ECOS_BB:
                    candidates['conic_solvers'].append(slv)

        # If gp we must have only conic solvers
        if gp:
            if solver is not None and solver not in slv_def.CONIC_SOLVERS:
                raise error.SolverError(
                  "When `gp=True`, `solver` must be a conic solver "
                  "(received '%s'); try calling " % solver +
                  " `solve()` with `solver=cvxpy.ECOS`."
                  )
            elif solver is None:
                candidates['qp_solvers'] = []  # No QP solvers allowed

        if self.is_mixed_integer():
            # ECOS_BB must be called explicitly.
            if slv_def.INSTALLED_MI_SOLVERS == [s.ECOS_BB] and solver != s.ECOS_BB:
                msg = """

                    You need a mixed-integer solver for this model. Refer to the documentation
                        https://www.cvxpy.org/tutorial/advanced/index.html#mixed-integer-programs
                    for discussion on this topic.

                    Quick fix 1: if you install the python package CVXOPT (pip install cvxopt),
                    then CVXPY can use the open-source mixed-integer solver `GLPK`.

                    Quick fix 2: you can explicitly specify solver='ECOS_BB'. This may result
                    in incorrect solutions and is not recommended.
                """
                raise error.SolverError(msg)
            candidates['qp_solvers'] = [
                s for s in candidates['qp_solvers']
                if slv_def.SOLVER_MAP_QP[s].MIP_CAPABLE]
            candidates['conic_solvers'] = [
                s for s in candidates['conic_solvers']
                if slv_def.SOLVER_MAP_CONIC[s].MIP_CAPABLE]
            if not candidates['conic_solvers'] and \
                    not candidates['qp_solvers']:
                raise error.SolverError(
                    "Problem is mixed-integer, but candidate "
                    "QP/Conic solvers (%s) are not MIP-capable." %
                    (candidates['qp_solvers'] +
                     candidates['conic_solvers']))

        return candidates

    def _add_custom_solver_candidates(self, custom_solver: Solver):
        """
        Returns a list of candidate solvers where custom_solver is the only potential option.

        Arguments
        ---------
        custom_solver : Solver

        Returns
        -------
        dict
            A dictionary of compatible solvers divided in `qp_solvers`
            and `conic_solvers`.

        Raises
        ------
        cvxpy.error.SolverError
            Raised if the name of the custom solver conflicts with the name of some officially
            supported solver
        """
        if custom_solver.name() in SOLVERS:
            message = "Custom solvers must have a different name than the officially supported ones"
            raise(error.SolverError(message))

        candidates = {'qp_solvers': [], 'conic_solvers': []}
        if not self.is_mixed_integer() or custom_solver.MIP_CAPABLE:
            if isinstance(custom_solver, QpSolver):
                SOLVER_MAP_QP[custom_solver.name()] = custom_solver
                candidates['qp_solvers'] = [custom_solver.name()]
            elif isinstance(custom_solver, ConicSolver):
                SOLVER_MAP_CONIC[custom_solver.name()] = custom_solver
                candidates['conic_solvers'] = [custom_solver.name()]
        return candidates

    def _construct_chain(self, solver=None, gp: bool = False, enforce_dpp: bool = False):
        """
        Construct the chains required to reformulate and solve the problem.

        In particular, this function

        # finds the candidate solvers
        # constructs the solving chain that performs the
           numeric reductions and solves the problem.

        Arguments
        ---------
        solver : str, optional
            The solver to use. Defaults to ECOS.
        gp : bool, optional
            If True, the problem is parsed as a Disciplined Geometric Program
            instead of as a Disciplined Convex Program.
        enforce_dpp : bool, optional
            Whether to error on DPP violations.

        Returns
        -------
        A solving chain
        """
        candidate_solvers = self._find_candidate_solvers(solver=solver, gp=gp)
        self._sort_candidate_solvers(candidate_solvers)
        return construct_solving_chain(self, candidate_solvers, gp=gp,
                                       enforce_dpp=enforce_dpp)

    @staticmethod
    def _sort_candidate_solvers(solvers) -> None:
        """Sorts candidate solvers lists according to slv_def.CONIC_SOLVERS/QP_SOLVERS

        Arguments
        ---------
        candidates : dict
            Dictionary of candidate solvers divided in qp_solvers
            and conic_solvers
        Returns
        -------
        None
        """
        if len(solvers['conic_solvers']) > 1:
            solvers['conic_solvers'] = sorted(
                solvers['conic_solvers'], key=lambda s: slv_def.CONIC_SOLVERS.index(s)
            )
        if len(solvers['qp_solvers']) > 1:
            solvers['qp_solvers'] = sorted(
                solvers['qp_solvers'], key=lambda s: slv_def.QP_SOLVERS.index(s)
            )

    def _invalidate_cache(self) -> None:
        self._cache_key = None
        self._solving_chain = None
        self._param_prog = None
        self._inverse_data = None

    def _solve(self,
               solver: str = None,
               warm_start: bool = True,
               verbose: bool = False,
               gp: bool = False,
               qcp: bool = False,
               requires_grad: bool = False,
               enforce_dpp: bool = False,
               **kwargs):
        """Solves a DCP compliant optimization problem.

        Saves the values of primal and dual variables in the variable
        and constraint objects, respectively.

        Arguments
        ---------
        solver : str, optional
            The solver to use. Defaults to ECOS.
        warm_start : bool, optional
            Should the previous solver result be used to warm start?
        verbose : bool, optional
            Overrides the default of hiding solver output.
        gp : bool, optional
            If True, parses the problem as a disciplined geometric program.
        qcp : bool, optional
            If True, parses the problem as a disciplined quasiconvex program.
        requires_grad : bool, optional
            Makes it possible to compute gradients with respect to
            parameters by calling `backward()` after solving, or to compute
            perturbations to the variables by calling `derivative()`. When
            True, the solver must be SCS, and dqcp must be False.
            A DPPError is thrown when problem is not DPP.
        enforce_dpp : bool, optional
            When True, a DPPError will be thrown when trying to solve a non-DPP
            problem (instead of just a warning). Defaults to False.
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
        if verbose:
            print(_HEADER)

        for parameter in self.parameters():
            if parameter.value is None:
                raise error.ParameterError(
                    "A Parameter (whose name is '%s') does not have a value "
                    "associated with it; all Parameter objects must have "
                    "values before solving a problem." % parameter.name())

        if verbose:
            n_variables = sum(np.prod(v.shape) for v in self.variables())
            n_parameters = sum(np.prod(p.shape) for p in self.parameters())
            s.LOGGER.info(
                    'Your problem has %d variables, '
                    '%d constraints, and ' '%d parameters.',
                    n_variables, len(self.constraints), n_parameters)
            curvatures = []
            if self.is_dcp():
                curvatures.append('DCP')
            if self.is_dgp():
                curvatures.append('DGP')
            if self.is_dqcp():
                curvatures.append('DQCP')
            s.LOGGER.info(
                    'It is compliant with the following grammars: %s',
                    ', '.join(curvatures))
            if n_parameters == 0:
                s.LOGGER.info(
                    '(If you need to solve this problem multiple times, '
                    'but with different data, consider using parameters.)')
            s.LOGGER.info(
                    'CVXPY will first compile your problem; then, it will '
                    'invoke a numerical solver to obtain a solution.')

        if requires_grad:
            dpp_context = 'dgp' if gp else 'dcp'
            if qcp:
                raise ValueError("Cannot compute gradients of DQCP problems.")
            elif not self.is_dpp(dpp_context):
                raise error.DPPError("Problem is not DPP (when requires_grad "
                                     "is True, problem must be DPP).")
            elif solver is not None and solver not in [s.SCS, s.DIFFCP]:
                raise ValueError("When requires_grad is True, the only "
                                 "supported solver is SCS "
                                 "(received %s)." % solver)
            elif s.DIFFCP not in slv_def.INSTALLED_SOLVERS:
                raise ImportError(
                    "The Python package diffcp must be installed to "
                    "differentiate through problems. Please follow the "
                    "installation instructions at "
                    "https://github.com/cvxgrp/diffcp")
            else:
                solver = s.DIFFCP
        else:
            if gp and qcp:
                raise ValueError("At most one of `gp` and `qcp` can be True.")
            if qcp and not self.is_dcp():
                if not self.is_dqcp():
                    raise error.DQCPError("The problem is not DQCP.")
                if verbose:
                    s.LOGGER.info(
                            'Reducing DQCP problem to a one-parameter '
                            'family of DCP problems, for bisection.')
                reductions = [dqcp2dcp.Dqcp2Dcp()]
                start = time.time()
                if type(self.objective) == Maximize:
                    reductions = [FlipObjective()] + reductions
                chain = Chain(problem=self, reductions=reductions)
                soln = bisection.bisect(
                    chain.reduce(), solver=solver, verbose=verbose, **kwargs)
                self.unpack(chain.retrieve(soln))
                return self.value

        data, solving_chain, inverse_data = self.get_problem_data(
            solver, gp, enforce_dpp, verbose)

        if verbose:
            print(_NUM_SOLVER_STR)
            s.LOGGER.info(
                    'Invoking solver %s  to obtain a solution.',
                    solving_chain.reductions[-1].name())
        start = time.time()
        solution = solving_chain.solve_via_data(
            self, data, warm_start, verbose, kwargs)
        end = time.time()
        self._solve_time = end - start
        self.unpack_results(solution, solving_chain, inverse_data)
        if verbose:
            print(_FOOTER)
            s.LOGGER.info('Problem status: %s', self.status)
            s.LOGGER.info('Optimal value: %.3e', self.value)
            s.LOGGER.info('Compilation took %.3e seconds', self._compilation_time)
            s.LOGGER.info(
                    'Solver (including time spent in interface) took '
                    '%.3e seconds', self._solve_time)
        return self.value

    def backward(self) -> None:
        """Compute the gradient of a solution with respect to Parameters.

        This method differentiates through the solution map of the problem,
        obtaining the gradient of a solution with respect to the Parameters.
        In other words, it calculates the sensitivities of the Parameters
        with respect to perturbations in the optimal Variable values. This
        can be useful for integrating CVXPY into automatic differentiation
        toolkits.

        ``backward()`` populates the ``gradient`` attribute of each Parameter
        in the problem as a side-effect. It can only be called after calling
        ``solve()`` with ``requires_grad=True``.

        Below is a simple example:

        ::

            import cvxpy as cp
            import numpy as np

            p = cp.Parameter()
            x = cp.Variable()
            quadratic = cp.square(x - 2 * p)
            problem = cp.Problem(cp.Minimize(quadratic), [x >= 0])
            p.value = 3.0
            problem.solve(requires_grad=True, eps=1e-10)
            # backward() populates the gradient attribute of the parameters
            problem.backward()
            # Because x* = 2 * p, dx*/dp = 2
            np.testing.assert_allclose(p.gradient, 2.0)

        In the above example, the gradient could easily be computed by hand.
        The ``backward()`` is useful because for almost all problems, the
        gradient cannot be computed analytically.

        This method can be used to differentiate through any DCP or DGP
        problem, as long as the problem is DPP compliant (i.e.,
        ``problem.is_dcp(dpp=True)`` or ``problem.is_dgp(dpp=True)`` evaluates to
        ``True``).

        This method uses the chain rule to evaluate the gradients of a
        scalar-valued function of the Variables with respect to the Parameters.
        For example, let x be a variable and p a Parameter; x and p might be
        scalars, vectors, or matrices. Let f be a scalar-valued function, with
        z = f(x). Then this method computes dz/dp = (dz/dx) (dx/p). dz/dx
        is chosen as the all-ones vector by default, corresponding to
        choosing f to be the sum function. You can specify a custom value for
        dz/dx by setting the ``gradient`` attribute on your variables. For example,

        ::

            import cvxpy as cp
            import numpy as np


            b = cp.Parameter()
            x = cp.Variable()
            quadratic = cp.square(x - 2 * b)
            problem = cp.Problem(cp.Minimize(quadratic), [x >= 0])
            b.value = 3.
            problem.solve(requires_grad=True, eps=1e-10)
            x.gradient = 4.
            problem.backward()
            # dz/dp = dz/dx dx/dp = 4. * 2. == 8.
            np.testing.assert_allclose(b.gradient, 8.)

        The ``gradient`` attribute on a variable can also be interpreted as a
        perturbation to its optimal value.

        Raises
        ------
            ValueError
                if solve was not called with ``requires_grad=True``
            SolverError
                if the problem is infeasible or unbounded
        """
        if s.DIFFCP not in self._solver_cache:
            raise ValueError("backward can only be called after calling "
                             "solve with `requires_grad=True`")
        elif self.status not in s.SOLUTION_PRESENT:
            raise error.SolverError("Backpropagating through "
                                    "infeasible/unbounded problems is not "
                                    "yet supported. Please file an issue on "
                                    "Github if you need this feature.")

        # TODO(akshayka): Backpropagate through dual variables as well.
        backward_cache = self._solver_cache[s.DIFFCP]
        DT = backward_cache["DT"]
        zeros = np.zeros(backward_cache["s"].shape)
        del_vars = {}

        gp = self._cache.gp()
        for variable in self.variables():
            if variable.gradient is None:
                del_vars[variable.id] = np.ones(variable.shape)
            else:
                del_vars[variable.id] = np.asarray(variable.gradient,
                                                   dtype=np.float64)
            if gp:
                # x_gp = exp(x_cone_program),
                # dx_gp/d x_cone_program = exp(x_cone_program) = x_gp
                del_vars[variable.id] *= variable.value

        dx = self._cache.param_prog.split_adjoint(del_vars)
        start = time.time()
        dA, db, dc = DT(dx, zeros, zeros)
        end = time.time()
        backward_cache['DT_TIME'] = end - start
        dparams = self._cache.param_prog.apply_param_jac(dc, -dA, db)

        if not gp:
            for param in self.parameters():
                param.gradient = dparams[param.id]
        else:
            dgp2dcp = self._cache.solving_chain.get(Dgp2Dcp)
            old_params_to_new_params = dgp2dcp.canon_methods._parameters
            for param in self.parameters():
                # Note: if param is an exponent in a power or gmatmul atom,
                # then the parameter passes through unchanged to the DCP
                # program; if the param is also used elsewhere (not as an
                # exponent), then param will also be in
                # old_params_to_new_params. Therefore, param.gradient =
                # dparams[param.id] (or 0) + 1/param*dparams[new_param.id]
                #
                # Note that param.id is in dparams if and only if
                # param was used as an exponent (because this means that
                # the parameter entered the DCP problem unchanged.)
                grad = 0.0 if param.id not in dparams else dparams[param.id]
                if param in old_params_to_new_params:
                    new_param = old_params_to_new_params[param]
                    # new_param.value == log(param), apply chain rule
                    grad += (1.0 / param.value) * dparams[new_param.id]
                param.gradient = grad

    def derivative(self) -> None:
        """Apply the derivative of the solution map to perturbations in the Parameters

        This method applies the derivative of the solution map to perturbations
        in the Parameters to obtain perturbations in the optimal values of the
        Variables. In other words, it tells you how the optimal values of the
        Variables would be changed by small changes to the Parameters.

        You can specify perturbations in a Parameter by setting its ``delta``
        attribute (if unspecified, the perturbation defaults to 0).

        This method populates the ``delta`` attribute of the Variables as a
        side-effect.

        This method can only be called after calling ``solve()`` with
        ``requires_grad=True``. It is compatible with both DCP and DGP
        problems (that are also DPP-compliant).

        Below is a simple example:

        ::

            import cvxpy as cp
            import numpy as np

            p = cp.Parameter()
            x = cp.Variable()
            quadratic = cp.square(x - 2 * p)
            problem = cp.Problem(cp.Minimize(quadratic), [x >= 0])
            p.value = 3.0
            problem.solve(requires_grad=True, eps=1e-10)
            # derivative() populates the delta attribute of the variables
            p.delta = 1e-3
            problem.derivative()
            # Because x* = 2 * p, dx*/dp = 2, so (dx*/dp)(p.delta) == 2e-3
            np.testing.assert_allclose(x.delta, 2e-3)

        Raises
        ------
            ValueError
                if solve was not called with ``requires_grad=True``
            SolverError
                if the problem is infeasible or unbounded
        """
        if s.DIFFCP not in self._solver_cache:
            raise ValueError("derivative can only be called after calling "
                             "solve with `requires_grad=True`")
        elif self.status not in s.SOLUTION_PRESENT:
            raise ValueError("Differentiating through infeasible/unbounded "
                             "problems is not yet supported. Please file an "
                             "issue on Github if you need this feature.")
        # TODO(akshayka): Forward differentiate dual variables as well
        backward_cache = self._solver_cache[s.DIFFCP]
        param_prog = self._cache.param_prog
        D = backward_cache["D"]
        param_deltas = {}

        gp = self._cache.gp()
        if gp:
            dgp2dcp = self._cache.solving_chain.get(Dgp2Dcp)

        if not self.parameters():
            for variable in self.variables():
                variable.delta = np.zeros(variable.shape)
            return

        for param in self.parameters():
            delta = param.delta if param.delta is not None else np.zeros(param.shape)
            if gp:
                if param in dgp2dcp.canon_methods._parameters:
                    new_param_id = dgp2dcp.canon_methods._parameters[param].id
                else:
                    new_param_id = param.id
                param_deltas[new_param_id] = (
                    1.0/param.value * np.asarray(delta, dtype=np.float64))
                if param.id in param_prog.param_id_to_col:
                    # here, param generated a new parameter and also
                    # passed through to the param cone prog unchanged
                    # (because it was an exponent of a power)
                    param_deltas[param.id] = np.asarray(delta,
                                                        dtype=np.float64)
            else:
                param_deltas[param.id] = np.asarray(delta, dtype=np.float64)
        dc, _, dA, db = param_prog.apply_parameters(param_deltas,
                                                    zero_offset=True)
        start = time.time()
        dx, _, _ = D(-dA, db, dc)
        end = time.time()
        backward_cache['D_TIME'] = end - start
        dvars = param_prog.split_solution(
            dx, [v.id for v in self.variables()])
        for variable in self.variables():
            variable.delta = dvars[variable.id]
            if gp:
                # x_gp = exp(x_cone_program),
                # dx_gp/d x_cone_program = exp(x_cone_program) = x_gp
                variable.delta *= variable.value

    def _clear_solution(self) -> None:
        for v in self.variables():
            v.save_value(None)
        for c in self.constraints:
            for dv in c.dual_variables:
                dv.save_value(None)
        self._value = None
        self._status = None
        self._solution = None

    def unpack(self, solution) -> None:
        """Updates the problem state given a Solution.

        Updates problem.status, problem.value and value of primal and dual
        variables. If solution.status is in cvxpy.settins.ERROR, this method
        is a no-op.

        Arguments
        _________
        solution : cvxpy.Solution
            A Solution object.

        Raises
        ------
        ValueError
            If the solution object has an invalid status
        """
        if solution.status in s.SOLUTION_PRESENT:
            for v in self.variables():
                v.save_value(solution.primal_vars[v.id])
            for c in self.constraints:
                if c.id in solution.dual_vars:
                    c.save_dual_value(solution.dual_vars[c.id])
            # Eliminate confusion of problem.value versus objective.value.
            self._value = self.objective.value
        elif solution.status in s.INF_OR_UNB:
            for v in self.variables():
                v.save_value(None)
            for constr in self.constraints:
                for dv in constr.dual_variables:
                    dv.save_value(None)
            self._value = solution.opt_val
        else:
            raise ValueError("Cannot unpack invalid solution: %s" % solution)

        self._status = solution.status
        self._solution = solution

    def unpack_results(self, solution, chain, inverse_data) -> None:
        """Updates the problem state given the solver results.

        Updates problem.status, problem.value and value of
        primal and dual variables.

        Arguments
        _________
        solution : object
            The solution returned by applying the chain to the problem
            and invoking the solver on the resulting data.
        chain : SolvingChain
            A solving chain that was used to solve the problem.
        inverse_data : list
            The inverse data returned by applying the chain to the problem.

        Raises
        ------
        cvxpy.error.SolverError
            If the solver failed
        """

        solution = chain.invert(solution, inverse_data)
        if solution.status in s.INACCURATE:
            warnings.warn(
                "Solution may be inaccurate. Try another solver, "
                "adjusting the solver settings, or solve with "
                "verbose=True for more information."
            )
        if solution.status in s.ERROR:
            raise error.SolverError(
                    "Solver '%s' failed. " % chain.solver.name() +
                    "Try another solver, or solve with verbose=True for more "
                    "information.")

        self.unpack(solution)
        self._solver_stats = SolverStats(self._solution.attr,
                                         chain.solver.name())

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

    def __repr__(self) -> str:
        return "Problem(%s, %s)" % (repr(self.objective),
                                    repr(self.constraints))

    def __neg__(self) -> "Problem":
        return Problem(-self.objective, self.constraints)

    def __add__(self, other):
        if other == 0:
            return self
        elif not isinstance(other, Problem):
            raise NotImplementedError()
        return Problem(self.objective + other.objective,
                       unique_list(self.constraints + other.constraints))

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            raise NotImplementedError()

    def __sub__(self, other):
        if not isinstance(other, Problem):
            raise NotImplementedError()
        return Problem(self.objective - other.objective,
                       unique_list(self.constraints + other.constraints))

    def __rsub__(self, other):
        if other == 0:
            return -self
        else:
            raise NotImplementedError()

    def __mul__(self, other):
        if not isinstance(other, (int, float)):
            raise NotImplementedError()
        return Problem(self.objective * other, self.constraints)

    __rmul__ = __mul__

    def __div__(self, other):
        if not isinstance(other, (int, float)):
            raise NotImplementedError()
        return Problem(self.objective * (1.0 / other), self.constraints)

    def is_constant(self) -> bool:
        return False

    __truediv__ = __div__


class SolverStats:
    """Reports some of the miscellaneous information that is returned
    by the solver after solving but that is not captured directly by
    the Problem instance.

    Attributes
    ----------
    solver_name : str
        The name of the solver.
    solve_time : double
        The time (in seconds) it took for the solver to solve the problem.
    setup_time : double
        The time (in seconds) it took for the solver to setup the problem.
    num_iters : int
        The number of iterations the solver had to go through to find a solution.
    extra_stats : object
        Extra statistics specific to the solver; these statistics are typically
        returned directly from the solver, without modification by CVXPY.
        This object may be a dict, or a custom Python object.
    """
    def __init__(self, results_dict, solver_name) -> None:
        self.solver_name = solver_name
        self.solve_time = None
        self.setup_time = None
        self.num_iters = None
        self.extra_stats = None

        if s.SOLVE_TIME in results_dict:
            self.solve_time = results_dict[s.SOLVE_TIME]
        if s.SETUP_TIME in results_dict:
            self.setup_time = results_dict[s.SETUP_TIME]
        if s.NUM_ITERS in results_dict:
            self.num_iters = results_dict[s.NUM_ITERS]
        if s.EXTRA_STATS in results_dict:
            self.extra_stats = results_dict[s.EXTRA_STATS]


class SizeMetrics:
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

    def __init__(self, problem) -> None:
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

            max_big_small_squared = float(big)*(float(small)**2)
            if self.max_big_small_squared < max_big_small_squared:
                self.max_big_small_squared = max_big_small_squared

        # num_scalar_eq_constr
        self.num_scalar_eq_constr = 0
        for constraint in problem.constraints:
            if isinstance(constraint, (Equality, Zero)):
                self.num_scalar_eq_constr += constraint.expr.size

        # num_scalar_leq_constr
        self.num_scalar_leq_constr = 0
        for constraint in problem.constraints:
            if isinstance(constraint, (Inequality, NonPos, NonNeg)):
                self.num_scalar_leq_constr += constraint.expr.size
