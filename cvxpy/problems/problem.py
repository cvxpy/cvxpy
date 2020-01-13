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
from cvxpy import error
from cvxpy.problems.objective import Minimize, Maximize
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.dqcp2dcp import dqcp2dcp
from cvxpy.reductions.eval_params import EvalParams
from cvxpy.reductions.flip_objective import FlipObjective
from cvxpy.reductions.solvers.solving_chain import construct_solving_chain
from cvxpy.interface.matrix_utilities import scalar_value
from cvxpy.reductions.solvers import bisection
from cvxpy.reductions.solvers import defines as slv_def
from cvxpy.utilities.deterministic import unique_list
import cvxpy.utilities.performance_utils as perf
from cvxpy.constraints import Equality, Inequality, NonPos, Zero
import cvxpy.utilities as u

from collections import namedtuple
import numpy as np


SolveResult = namedtuple(
    'SolveResult',
    ['opt_value', 'status', 'primal_values', 'dual_values'])


class Cache(object):
    def __init__(self):
        self.key = None
        self.solving_chain = None
        self.param_cone_prog = None
        self.inverse_data = None

    def invalidate(self):
        self.key = None
        self.solving_chain = None
        self.param_cone_prog = None
        self.inverse_data = None

    def make_key(self, solver, gp):
        return (solver, gp)


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
            raise error.DCPError("Problem objective must be Minimize or Maximize.")
        # Constraints and objective are immutable.
        self._objective = objective
        self._constraints = [c for c in constraints]
        self._value = None
        self._status = None
        self._solution = None
        self._cache = Cache()
        self._solver_cache = {}
        # Information about the shape of the problem and its constituent parts
        self._size_metrics = None
        # Benchmarks reported by the solver:
        self._solver_stats = None
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

    @perf.compute_once
    def is_dcp(self):
        """Does the problem satisfy DCP rules?
        """
        return all(
          expr.is_dcp() for expr in self.constraints + [self.objective])

    @perf.compute_once
    def is_dpp(self):
        """Does the problem satisfy DPP rules?
        """
        return all(
          expr.is_dpp() for expr in self.constraints + [self.objective])

    @perf.compute_once
    def is_dgp(self):
        """Does the problem satisfy DGP rules?
        """
        return all(
          expr.is_dgp() for expr in self.constraints + [self.objective])

    @perf.compute_once
    def is_dqcp(self):
        """Does the problem satisfy the DQCP rules?
        """
        return all(
          expr.is_dqcp() for expr in self.constraints + [self.objective])

    @perf.compute_once
    def is_qp(self):
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
    def is_mixed_integer(self):
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
        """Solves the problem using the specified method.

        Populates the :code:`status` and :code:`value` attributes on the
        problem object as a side-effect.

        Parameters
        ----------
        solver : str, optional
            The solver to use. For example, 'ECOS', 'SCS', or 'OSQP'.
        verbose : bool, optional
            Overrides the default of hiding solver output.
        gp : bool, optional
            If True, parses the problem as a disciplined geometric program
            instead of a disciplined convex program.
        qcp : bool, optional
            If True, parses the problem as a disciplined quasiconvex program
            instead of a disciplined convex program.
        requires_grad : bool, optional
            Makes it possible to compute gradients with respect to
            parameters by calling `.backward()` after solving, or to compute
            perturbations to the variables by calling `.derivative()`. When
            True, the solver must be SCS, and gp, qcp must be false; a DPPError is
            thrown when problem is not DPP.
        method : function, optional
            A custom solve method to use.
        kwargs : keywords, optional
            Additional solver specific arguments. See Notes below.

        Notes
        ------
        CVXPY interfaces with a wide range of solvers; the algorithms used by these solvers
        have parameters relating to stopping criteria, and strategies to improve solution quality.

        There is no one choice of parameters which is perfect for every problem. If you are not
        getting satisfactory results from a solver, you can try changing its parameters. The
        exact way this is done depends on the specific solver. Here are some examples:

            prob.solve(solver='ECOS', abstol=1e-6)
            prob.solve(solver='OSQP', max_iter=10000).
            mydict = {"MSK_DPAR_INTPNT_CO_TOL_NEAR_REL":  10}
            prob.solve(solver='MOSEK', mosek_params=mydict).

        You should refer to CVXPY's web documentation for details on how to pass solver
        parameters.

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

    def get_problem_data(self, solver, gp=False):
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

        Parameters
        ----------
        solver : str
            The solver the problem data is for.
        gp : bool, optional
            If True, then parses the problem as a disciplined geometric program
            instead of a disciplined convex program.

        Returns
        -------
        dict or object
            lowest level representation of problem
        SolvingChain
            The solving chain that created the data.
        list
            The inverse data generated by the chain.
        """
        key = self._cache.make_key(solver, gp)
        if key != self._cache.key:
            self._cache.invalidate()
            solving_chain = self._construct_chain(solver=solver, gp=gp)
            self._cache.key = key
            self._cache.solving_chain = solving_chain
            self._solver_cache = {}
        else:
            solving_chain = self._cache.solving_chain

        if self._cache.param_cone_prog is not None:
            # fast path, bypasses application of reductions
            data, solver_inverse_data = solving_chain.solver.apply(
                self._cache.param_cone_prog)
            inverse_data = self._cache.inverse_data + [solver_inverse_data]
        else:
            data, inverse_data = solving_chain.apply(self)
            safe_to_cache = (
                isinstance(data, dict)
                and s.PARAM_PROB in data
                and not any(isinstance(reduction, EvalParams)
                            for reduction in solving_chain.reductions)
            )
            if safe_to_cache:
                self._cache.param_cone_prog = data[s.PARAM_PROB]
                # the last datum in inverse_data corresponds to the solver,
                # so we shouldn't cache it
                self._cache.inverse_data = inverse_data[:-1]
        return data, solving_chain, inverse_data

    def _find_candidate_solvers(self,
                                solver=None,
                                gp=False):
        """
        Find candiate solvers for the current problem. If solver
        is not None, it checks if the specified solver is compatible
        with the problem passed.

        Parameters
        ----------
        solver : string
            The name of the solver with which to solve the problem. If no
            solver is supplied (i.e., if solver is None), then the targeted
            solver may be any of those that are installed. If the problem
            is variable-free, then this parameter is ignored.
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
            candidates['conic_solvers'] = [s for s in slv_def.INSTALLED_SOLVERS
                                           if s in slv_def.CONIC_SOLVERS]

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
                    [candidates['qp_solvers'], candidates['conic_solvers']])

        return candidates

    def _construct_chain(self, solver=None, gp=False):
        """
        Construct the chains required to reformulate and solve the problem.

        In particular, this function

        # finds the candidate solvers
        # constructs the solving chain that performs the
           numeric reductions and solves the problem.

        Parameters
        ----------
        solver : str, optional
            The solver to use. Defaults to ECOS.
        gp : bool, optional
            If True, the problem is parsed as a Disciplined Geometric Program
            instead of as a Disciplined Convex Program.

        Returns
        -------
        A solving chain
        """
        candidate_solvers = self._find_candidate_solvers(solver=solver, gp=gp)
        return construct_solving_chain(self, candidate_solvers, gp=gp)

    def _invalidate_cache(self):
        self._cache_key = None
        self._solving_chain = None
        self._param_cone_prog = None
        self._inverse_data = None

    def _solve(self,
               solver=None,
               warm_start=True,
               verbose=False,
               gp=False, qcp=False, requires_grad=False, **kwargs):
        """Solves a DCP compliant optimization problem.

        Saves the values of primal and dual variables in the variable
        and constraint objects, respectively.

        Parameters
        ----------
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
            parameters by calling `.backward()` after solving, or to compute
            perturbations to the variables by calling `.derivative()`. When
            True, the solver must be SCS, and gp, qcp must be False;
            a DPPError is thrown when problem is not DPP.
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
        for parameter in self.parameters():
            if parameter.value is None:
                raise error.ParameterError(
                    "A Parameter (whose name is '%s') does not have a value "
                    "associated with it; all Parameter objects must have "
                    "values before solving a problem." % parameter.name())

        if requires_grad:
            if not self.is_dpp():
                raise error.DPPError("Problem is not DPP (when requires_grad "
                                     "is True, problem must be DPP).")
            elif gp:
                raise ValueError("Cannot compute gradients of DGP problems.")
            elif qcp:
                raise ValueError("Cannot compute gradients of DQCP problems.")
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
                reductions = [dqcp2dcp.Dqcp2Dcp()]
                if type(self.objective) == Maximize:
                    reductions = [FlipObjective()] + reductions
                chain = Chain(problem=self, reductions=reductions)
                soln = bisection.bisect(
                    chain.reduce(), solver=solver, verbose=verbose, **kwargs)
                self.unpack(chain.retrieve(soln))
                return self.value

        data, solving_chain, inverse_data = self.get_problem_data(solver, gp)
        solution = solving_chain.solve_via_data(
            self, data, warm_start, verbose, kwargs)
        self.unpack_results(solution, solving_chain, inverse_data)
        return self.value

    def backward(self):
        """Compute the gradient of a solution with respect to parameters.

        This method differentiates through the solution map of the problem,
        to obtain the gradient of a solution with respect to the parameters.
        In other words, it calculates the sensitivities of the parameters
        with respect to perturbations in the optimal variable values.

        .backward() populates the .gradient attribute of each parameter as a
        side-effect. It can only be called after calling .solve() with
        `requires_grad=True`.

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
            # .backward() populates the .gradient attribute of the parameters
            problem.backward()
            # Because x* = 2 * p, dx*/dp = 2
            np.testing.assert_allclose(p.gradient, 2.0)

        In the above example, the gradient could easily be computed by hand;
        however, .backward() can be used to differentiate through any DCP
        program (that is also DPP-compliant).

        This method uses the chain rule to evaluate the gradients of a
        scalar-valued function of the variables with respect to the parameters.
        For example, let x be a variable and p a parameter; x and p might be
        scalars, vectors, or matrices. Let f be a scalar-valued function, with
        z = f(x). Then this method computes dz/dp = (dz/dx) (dx/p). dz/dx
        is chosen to be the all ones vector by default, corresponding to
        choosing f to be the sum function. You can specify a custom value for
        dz/dx by setting the .gradient attribute on your variables. For example,

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

        The .gradient attribute on a variable can also be interpreted as a
        perturbation to its optimal value.

        Raises
        ------
            ValueError
                if solve was not called with `requires_grad=True`
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
        for variable in self.variables():
            if variable.gradient is None:
                del_vars[variable.id] = np.ones(variable.shape)
            else:
                del_vars[variable.id] = np.asarray(variable.gradient,
                                                   dtype=np.float64)
        dx = self._cache.param_cone_prog.split_adjoint(del_vars)
        dA, db, dc = DT(dx, zeros, zeros)
        dparams = self._cache.param_cone_prog.apply_param_jac(dc, -dA, db)
        for parameter in self.parameters():
            parameter.gradient = dparams[parameter.id]

    def derivative(self):
        """Apply the derivative of the solution map to perturbations in the parameters

        This method applies the derivative of the solution map to perturbations
        in the parameters, to obtain perturbations in the optimal values of the
        variables. In other words, it tells you how the optimal values of the
        variables would be changed.

        You can specify perturbations in a parameter by setting its .delta
        attribute (if unspecified, the perturbation defaults to 0). This method
        populates the .delta attribute of the variables as a side-effect.

        This method can only be called after calling .solve() with
        `requires_grad=True`.

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
            # .derivative() populates the .gradient attribute of the parameters
            problem.backward()
            p.delta = 1e-3
            # Because x* = 2 * p, dx*/dp = 2, so (dx*/dp)(p.delta) == 2e-3
            np.testing.assert_allclose(p.gradient, 2e-3)

        Raises
        ------
            ValueError
                if solve was not called with `requires_grad=True`
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
        param_cone_prog = self._cache.param_cone_prog
        D = backward_cache["D"]
        param_deltas = {}
        for parameter in self.parameters():
            if parameter.delta is None:
                param_deltas[parameter.id] = np.zeros(parameter.shape)
            else:
                param_deltas[parameter.id] = np.asarray(parameter.delta,
                                                        dtype=np.float64)
        dc, _, dA, db = param_cone_prog.apply_parameters(param_deltas,
                                                         zero_offset=True)
        dx, _, _ = D(-dA, db, dc)
        dvars = param_cone_prog.split_solution(
            dx, [v.id for v in self.variables()])
        for variable in self.variables():
            variable.delta = dvars[variable.id]

    def _clear_solution(self):
        for v in self.variables():
            v.save_value(None)
        for c in self.constraints:
            for dv in c.dual_variables:
                dv.save_value(None)
        self._value = None
        self._status = None
        self._solution = None

    def unpack(self, solution):
        """Updates the problem state given a Solution.

        Updates problem.status, problem.value and value of primal and dual
        variables. If solution.status is in cvxpy.settins.ERROR, this method
        is a no-op.

        Parameters
        __________
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
        elif solution.status in s.INF_OR_UNB:
            for v in self.variables():
                v.save_value(None)
            for constr in self.constraints:
                for dv in constr.dual_variables:
                    dv.save_value(None)
        else:
            raise ValueError("Cannot unpack invalid solution: %s" % solution)

        self._value = solution.opt_val
        self._status = solution.status
        self._solution = solution

    def unpack_results(self, solution, chain, inverse_data):
        """Updates the problem state given the solver results.

        Updates problem.status, problem.value and value of
        primal and dual variables.

        Parameters
        __________
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
                       unique_list(self.constraints + other.constraints))

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return NotImplemented

    def __sub__(self, other):
        if not isinstance(other, Problem):
            return NotImplemented
        return Problem(self.objective - other.objective,
                       unique_list(self.constraints + other.constraints))

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

    def is_constant(self):
        return False

    __truediv__ = __div__


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
            if isinstance(constraint, (Inequality, NonPos)):
                self.num_scalar_leq_constr += constraint.expr.size
