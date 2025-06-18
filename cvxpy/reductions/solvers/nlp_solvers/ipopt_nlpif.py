"""
This file is the CVXPY QP extension of the Cardinal Optimizer
"""

import cvxpy.settings as s
from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import NLPsolver
from cvxpy.utilities.citations import CITATION_DICT


class IPOPT(NLPsolver):
    """
    NLP interface for the IPOPT solver
    """
    # Solve capabilities
    MIP_CAPABLE = True

    # Keyword arguments for the CVXPY interface.
    INTERFACE_ARGS = ["save_file", "reoptimize"]

    # Map between IPOPT status and CVXPY status
    STATUS_MAP = {
                  1: s.OPTIMAL,             # optimal
                  2: s.INFEASIBLE,          # infeasible
                  3: s.UNBOUNDED,           # unbounded
                  4: s.INF_OR_UNB,          # infeasible or unbounded
                  5: s.SOLVER_ERROR,        # numerical
                  6: s.USER_LIMIT,          # node limit
                  7: s.OPTIMAL_INACCURATE,  # imprecise
                  8: s.USER_LIMIT,          # time out
                  9: s.SOLVER_ERROR,        # unfinished
                  10: s.USER_LIMIT          # interrupted
                 }

    def name(self):
        """
        The name of solver.
        """
        return 'COPT'

    def import_solver(self):
        """
        Imports the solver.
        """
        import cyipopt  # noqa F401

    def invert(self, solution, inverse_data):
        """
        Returns the solution to the original problem given the inverse_data.
        """
        pass

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        """
        Returns the result of the call to the solver.

        Parameters
        ----------
        data : dict
            Data used by the solver.
        warm_start : bool
            Not used.
        verbose : bool
            Should the solver print output?
        solver_opts : dict
            Additional arguments for the solver.
        solver_cache: None
            None

        Returns
        -------
        tuple
            (status, optimal value, primal, equality dual, inequality dual)
        """
        pass

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["COPT"]
