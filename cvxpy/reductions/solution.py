"""
Copyright 2013 Steven Diamond

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
import pickle

import numpy as np

import cvxpy.settings as s

INF_OR_UNB_MESSAGE = """
    The problem is either infeasible or unbounded, but the solver
    cannot tell which. Disable any solver-specific presolve methods
    and re-solve to determine the precise problem status.

    For GUROBI and CPLEX you can automatically perform this re-solve
    with the keyword argument prob.solve(reoptimize=True, ...).
    """


def failure_solution(status, attr=None, dual_vars=None) -> "Solution":
    """Factory function for infeasible or unbounded solutions.

    Parameters
    ----------
    status : str
        The problem status.
    attr : dict
        Miscellaneous information propagated up from a solver.
    dual_vars : dict of id to NumPy ndarray
        A map from constraint ids to dual values.

    Returns
    -------
    Solution
        A solution object.
    """
    if status in [s.INFEASIBLE, s.INFEASIBLE_INACCURATE]:
        opt_val = np.inf
    elif status in [s.UNBOUNDED, s.UNBOUNDED_INACCURATE]:
        opt_val = -np.inf
    else:
        opt_val = None
    if attr is None:
        attr = {}
    if dual_vars is None:
        dual_vars = {}
    if status == s.INFEASIBLE_OR_UNBOUNDED:
        attr['message'] = INF_OR_UNB_MESSAGE
    return Solution(status, opt_val, {}, dual_vars, attr)


class Solution:
    """A solution to an optimization problem.

    Attributes
    ----------
    status : str
        The status code.
    opt_val : float
        The optimal value.
    primal_vars : dict of id to NumPy ndarray
        A map from variable ids to optimal values.
    dual_vars : dict of id to NumPy ndarray
        A map from constraint ids to dual values.
    attr : dict
        Miscellaneous information propagated up from a solver.
    """
    def __init__(self, status, opt_val, primal_vars, dual_vars, attr) -> None:
        self.status = status
        self.opt_val = opt_val
        self.primal_vars = primal_vars
        self.dual_vars = dual_vars
        self.attr = attr

    def copy(self) -> "Solution":
        return Solution(self.status,
                        self.opt_val,
                        self.primal_vars,
                        self.dual_vars,
                        self.attr)

    def __str__(self) -> str:
        return "Solution(status=%s, opt_val=%s, primal_vars=%s, dual_vars=%s, attr=%s)" % (
          self.status, self.opt_val, self.primal_vars, self.dual_vars, self.attr)

    def __repr__(self) -> str:
        return "Solution(%s, %s, %s, %s)" % (self.status,
                                             self.primal_vars,
                                             self.dual_vars,
                                             self.attr)

    def __getstate__(self) -> dict:
        """Return a pickleable state dict for Solution.

        Entries in ``attr`` that cannot be pickled are omitted to allow
        serialization of the solution.
        """
        state = self.__dict__.copy()
        if "attr" in state and isinstance(state["attr"], dict):
            clean_attr = {}
            for k, v in state["attr"].items():
                # OSQP results wrap info in a SimpleNamespace containing an unpicklable
                # _pybind11_conduit_v1_ bound method; strip it lazily upon serialization.
                if k == s.EXTRA_STATS and hasattr(v, "info") and hasattr(
                    v.info, "_pybind11_conduit_v1_"
                ):
                    try:
                        delattr(v.info, "_pybind11_conduit_v1_")
                    except (AttributeError, TypeError):
                        pass
                try:
                    pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL)
                    clean_attr[k] = v
                except (pickle.PicklingError, TypeError, AttributeError):
                    s.LOGGER.debug(
                        "Dropping unpicklable Solution.attr[%r] of type %s during serialization.",
                        k,
                        type(v).__name__,
                    )
            state["attr"] = clean_attr
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if "attr" not in self.__dict__ or self.attr is None:
            self.attr = {}
        if "primal_vars" not in self.__dict__ or self.primal_vars is None:
            self.primal_vars = {}
        if "dual_vars" not in self.__dict__ or self.dual_vars is None:
            self.dual_vars = {}
