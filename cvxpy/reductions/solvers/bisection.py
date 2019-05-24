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
import cvxpy.error as error
import cvxpy.problems as problems
from cvxpy.problems.objective import Minimize
from cvxpy.reductions import solution as solution_module


def _lower_problem(problem):
    """Evaluates lazy constraints."""
    constrs = [c() if callable(c) else c for c in problem.constraints]
    constrs = [c for c in constrs if c is not None]
    if s.INFEASIBLE in constrs:
        # Indicates that the problem is infeasible.
        return None
    return problems.problem.Problem(Minimize(0), constrs)


def _solve(problem, solver):
    if problem is None:
        return
    problem.solve(solver=solver)


def _infeasible(problem):
    return problem is None or problem.status in (s.INFEASIBLE,
                                                 s.INFEASIBLE_INACCURATE)


def _find_bisection_interval(problem, t, solver=None, low=None, high=None,
                             max_iters=100):
    """Finds an interval for bisection."""
    if low is None:
        low = 0 if t.is_nonneg() else -1
    if high is None:
        high = 0 if t.is_nonpos() else 1

    infeasible_low = t.is_nonneg()
    feasible_high = t.is_nonpos()
    for _ in range(max_iters):
        if not feasible_high:
            t.value = high
            lowered = _lower_problem(problem)
            _solve(lowered, solver)
            if _infeasible(lowered):
                low = high
                high *= 2
                continue
            elif lowered.status in s.SOLUTION_PRESENT:
                feasible_high = True
            else:
                raise error.SolverError(
                    "Solver failed with status %s" % lowered.status)

        if not infeasible_low:
            t.value = low
            lowered = _lower_problem(problem)
            _solve(lowered, solver=solver)
            if _infeasible(lowered):
                infeasible_low = True
            elif lowered.status in s.SOLUTION_PRESENT:
                high = low
                low *= 2
                continue
            else:
                raise error.SolverError(
                    "Solver failed with status %s" % lowered.status)

        if infeasible_low and feasible_high:
            return low, high

    raise error.SolverError("Unable to find suitable interval for bisection.")


def _bisect(problem, solver, t, low, high, tighten_lower, tighten_higher,
            eps=1e-6, verbose=False, max_iters=100):
    """Bisect `problem` on the parameter `t`."""

    verbose_freq = 5
    soln = None
    for i in range(max_iters):
        assert low <= high
        if soln is not None and (high - low) <= eps:
            # the previous iteration might have been infeasible, but
            # the tigthen* functions might have narrowed the interval
            # to the optimal value in the previous iteration (hence the
            # soln is not None check)
            return soln, low, high
        query_pt = (low + high) / 2.0
        if verbose and i % verbose_freq == 0:
            print("(iteration %d) lower bound: %0.6f" % (i, low))
            print("(iteration %d) upper bound: %0.6f" % (i, high))
            print("(iteration %d) query point: %0.6f " % (i, query_pt))
        t.value = query_pt
        lowered = _lower_problem(problem)
        _solve(lowered, solver=solver)

        if _infeasible(lowered):
            if verbose and i % verbose_freq == 0:
                print("(iteration %d) query was infeasible.\n" % i)
            low = tighten_lower(query_pt)
        elif lowered.status in s.SOLUTION_PRESENT:
            if verbose and i % verbose_freq == 0:
                print("(iteration %d) query was feasible. %s)\n" %
                      (i, lowered.solution))
            soln = lowered.solution
            high = tighten_higher(query_pt)
        else:
            if verbose:
                print("Aborting; the solver failed ...\n")
            raise error.SolverError(
                "Solver failed with status %s" % lowered.status)
    raise error.SolverError("Max iters hit during bisection.")


def bisect(problem, solver=None, low=None, high=None, eps=1e-6, verbose=False,
           max_iters=100, max_iters_interval_search=100):
    """Bisection on a one-parameter family of DCP problems.

    Bisects on a one-parameter family of DCP problems emitted by `Dqcp2Dcp`.

    Parameters
    ------
    problem : Problem
        problem emitted by Dqcp2Dcp
    solver : Solver
        solver to use for bisection
    low : float
        lower bound for bisection (optional)
    high : float
        upper bound for bisection (optional)
    eps : float
        terminate bisection when width of interval is < eps
    verbose : bool
        whether to print verbose output related to the bisection
    max_iters : int
        the maximum number of iterations to run bisection

    Returns
    -------
    A Solution object.
    """
    if not hasattr(problem, '_bisection_data'):
        raise ValueError("`bisect` only accepts problems emitted by Dqcp2Dcp.")

    feas_problem, t, tighten_lower, tighten_higher = problem._bisection_data
    if verbose:
        print("\n******************************************************"
              "**************************\n"
              "Preparing to bisect problem\n\n%s\n" % _lower_problem(problem))

    lowered_feas = _lower_problem(feas_problem)
    _solve(lowered_feas, solver)
    if _infeasible(lowered_feas):
        if verbose:
            print("Problem is infeasible.")
        return solution_module.failure_solution(s.INFEASIBLE)

    if low is None or high is None:
        if verbose:
            print("Finding interval for bisection ...")
        low, high = _find_bisection_interval(problem, t, solver, low, high,
                                             max_iters_interval_search)
    if verbose:
        print("initial lower bound: %0.6f" % low)
        print("initial upper bound: %0.6f\n" % high)

    soln, low, high = _bisect(
        problem, solver, t, low, high, tighten_lower, tighten_higher,
        eps, verbose, max_iters)
    soln.opt_val = (low + high) / 2.0
    if verbose:
        print("Bisection completed, with lower bound %0.6f and upper bound "
              "%0.7f\n******************************************"
              "**************************************\n"
              % (low, high))
    return soln
