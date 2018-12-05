"""
Copyright 2018 Akshay Agrawal

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
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.dgp2dcp.atom_canonicalizers import DgpCanonMethods

import numpy as np


class Dgp2Dcp(Canonicalization):
    """Reduce DGP problems to DCP problems.

    This reduction takes as input a DGP problem and returns an equivalent DCP
    problem. Because every (generalized) geometric program is a DGP problem,
    this reduction can be used to convert geometric programs into convex form.

    Example
    -------

    >>> import cvxpy as cp
    >>>
    >>> x1 = cp.Variable(pos=True)
    >>> x2 = cp.Variable(pos=True)
    >>> x3 = cp.Variable(pos=True)
    >>>
    >>> monomial = 3.0 * x_1**0.4 * x_2 ** 0.2 * x_3 ** -1.4
    >>> posynomial = monomial + 2.0 * x_1 * x_2
    >>> problem = cp.Problem(cp.Minimize(posynomial), [monomial == 4.0])
    >>>
    >>> dcp2cone = cvxpy.reductions.Dcp2Cone()
    >>> assert not dcp2cone.accepts(problem)
    >>>
    >>> gp2dcp = cvxpy.reductions.Dgp2Dcp()
    >>> assert gp2dcp.accepts(problem)
    >>>
    >>> dcp_problem = gp2dcp.reduce(problem)
    >>> assert dcp2cone.accepts(dcp_problem)
    >>> dcp_probem.solve()
    >>> gp_problem.unpack(gp2dcp.retrieve(dcp_problem.solution))
    >>>
    >>> print(gp_problem.value)
    >>> print(gp_problem.variables())
    """
    def accepts(self, problem):
        """A problem is accepted if it is DGP.
        """
        return problem.is_dgp()

    def apply(self, problem):
        """Converts a DGP problem to a DCP problem.
        """
        if not self.accepts(problem):
            raise ValueError("The supplied problem is not DGP.")

        self.canon_methods = DgpCanonMethods()
        equiv_problem, inverse_data = super(Dgp2Dcp, self).apply(problem)
        inverse_data._problem = problem
        return equiv_problem, inverse_data

    def invert(self, solution, inverse_data):
        solution = super(Dgp2Dcp, self).invert(solution, inverse_data)
        for vid, value in solution.primal_vars.items():
            solution.primal_vars[vid] = np.exp(value)
        # We unpack the solution in order to obtain the objective value in
        # terms of the original variables.
        if solution.status in s.SOLUTION_PRESENT:
            inverse_data._problem.unpack(solution)
            solution = inverse_data._problem.solution
            solution.opt_val = inverse_data._problem.objective.value
            inverse_data._problem._clear_solution()
        elif solution.status in s.INF_OR_UNB:
            solution.opt_val = np.exp(solution.opt_val)
        return solution
