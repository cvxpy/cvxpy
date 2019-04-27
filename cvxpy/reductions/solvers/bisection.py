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


def _find_bisection_interval(problem, t, low=None, high=None):
    if low is None:
        low = -1.0
    if high is None:
        high = 1.0

    for _ in range(100):
        t.value = high
        problem.solve()
        infeasible_low = False
        feasible_high = False
        if problem.status in (s.INFEASIBLE, s.INFEASIBLE_INACCURATE):
            low = high
            high *= 2
            continue
        elif problem.status in s.SOLUTION_PRESENT:
            feasible_high = True
        else:
            raise error.SolverError(
                "Solver failed with status %s" % problem.status)

        t.value = low
        problem.solve()
        if problem.status in s.SOLUTION_PRESENT:
            high = low
            low *= 2
            continue
        elif problem.status in (s.INFEASIBLE, s.INFEASIBLE_INACCURATE):
            infeasible_low = True

        if infeasible_low and feasible_high:
            return low, high

    raise error.SolverError("Unable to find suitable interval for bisection.")


def _bisect(problem, t, low, high, tighten_lower, tighten_higher, eps=1e-4,
            verbose=False, max_iters=100):
    """Bisect `problem` on the parameter `p`."""

    soln = None
    for i in range(max_iters):
        if low > high or (high - low) <= eps:
            return soln, low, high
        query_pt = (low + high) / 2.0
        if verbose and i % 5 == 0:
            print("(iteration %d) lower bound: %0.6f" % (i, low))
            print("(iteration %d) upper bound: %0.6f" % (i, high))
            print("(iteration %d) query point: %0.6f\n " % (i, query_pt))
        t.value = query_pt
        problem.solve()

        if problem.status in (s.INFEASIBLE, s.INFEASIBLE_INACCURATE):
            low = tighten_lower(query_pt)
        elif problem.status in s.SOLUTION_PRESENT:
            soln = problem.solution
            high = tighten_higher(query_pt)
        else:
            if verbose:
                print("Aborting; the solver failed ...\n")
            raise error.SolverError(
                "Solver failed with status %s" % problem.status)
    raise error.SolverError("Max iters hit during bisection.")


def bisect(bisection_data, low=None, high=None, eps=1e-6, verbose=False,
           max_iters=100):
    problem, t, tighten_lower, tighten_higher = bisection_data
    if verbose:
        print("\n******************************************************"
              "**************************\n"
              "Preparing to bisect problem\n\n%s\n" % problem)

    if low is None or high is None:
        if verbose:
            print("Finding interval for bisection ...")
        low, high = _find_bisection_interval(problem, t, low, high)
    if verbose:
        print("initial lower bound: %0.6f" % low)
        print("initial upper bound: %0.6f\n" % high)

    soln, low, high = _bisect(
        problem, t, low, high, tighten_lower, tighten_higher,
        eps, verbose, max_iters)
    if verbose:
        print("Bisection completed, with lower bound %0.6f and upper bound "
              "%0.7f\n******************************************"
              "**************************************\n"
              % (low, high))
    return soln
