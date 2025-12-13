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
import numpy as np

from cvxpy import settings
from cvxpy.expressions.expression import Expression
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.dgp2dcp.canonicalizers import DgpCanonMethods


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
    >>> dgp_problem = cp.Problem(cp.Minimize(posynomial), [monomial == 4.0])
    >>>
    >>> dcp2cone = cvxpy.reductions.Dcp2Cone()
    >>> assert not dcp2cone.accepts(dgp_problem)
    >>>
    >>> gp2dcp = cvxpy.reductions.Dgp2Dcp(dgp_problem)
    >>> dcp_problem = gp2dcp.reduce()
    >>>
    >>> assert dcp2cone.accepts(dcp_problem)
    >>> dcp_problem.solve()
    >>>
    >>> dgp_problem.unpack(gp2dcp.retrieve(dcp_problem.solution))
    >>> print(dgp_problem.value)
    >>> print(dgp_problem.variables())
    """
    def __init__(self, problem=None) -> None:
        # Canonicalization of DGP is stateful; canon_methods created
        # in `apply`.
        super(Dgp2Dcp, self).__init__(canon_methods=None, problem=problem)

    def accepts(self, problem):
        """A problem is accepted if it is DGP.
        """
        return problem.is_dgp()

    def update_parameters(self, problem) -> None:
        """Update log-parameter values from original parameters.

        Called at solve time in the DPP fast path. Parameters are transformed
        to log-space during canonicalization; this method sets the log-parameter
        values from the original parameter values.
        """
        if self.canon_methods is None:
            return
        for param in problem.parameters():
            if param in self.canon_methods._parameters:
                self.canon_methods._parameters[param].value = np.log(param.value)

    def param_backward(self, param, dparams):
        """Apply chain rule for log transformation in backward diff.

        For DGP, param -> log(param), so d(loss)/d(param) = d(loss)/d(log_param) / param.
        """
        if self.canon_methods is None:
            return None
        if param not in self.canon_methods._parameters:
            return None
        new_param = self.canon_methods._parameters[param]
        # Apply chain rule: d(log(x))/dx = 1/x
        return (1.0 / param.value) * dparams[new_param.id]

    def param_forward(self, param, delta):
        """Apply chain rule for log transformation in forward diff.

        For DGP, param -> log(param), so d(log_param) = d(param) / param.
        """
        if self.canon_methods is None:
            return None
        if param not in self.canon_methods._parameters:
            return None
        new_param = self.canon_methods._parameters[param]
        return {new_param.id: (1.0 / param.value) * np.asarray(delta, dtype=np.float64)}

    def var_backward(self, var, value):
        """Apply chain rule for exp transformation in backward diff.

        For DGP, x_gp = exp(x_cone), so dx_gp/dx_cone = exp(x_cone) = x_gp.
        """
        return value * var.value

    def var_forward(self, var, value):
        """Apply chain rule for exp transformation in forward diff.

        For DGP, x_gp = exp(x_cone), so dx_gp/dx_cone = exp(x_cone) = x_gp.
        """
        return value * var.value

    def apply(self, problem):
        """Converts a DGP problem to a DCP problem.
        """
        if not self.accepts(problem):
            raise ValueError("The supplied problem is not DGP.")

        self.canon_methods = DgpCanonMethods()
        equiv_problem, inverse_data = super(Dgp2Dcp, self).apply(problem)
        inverse_data._problem = problem
        return equiv_problem, inverse_data

    def canonicalize_expr(
            self, 
            expr: Expression, 
            args: list, 
            canonicalize_params: bool = True
        ):
        """Canonicalize an expression, w.r.t. canonicalized arguments.
        
        Args:
            expr: Expression to canonicalize.
            args: Arguments to the expression.
            canonicalize_params: Should constant subtrees 
                containing parameters be canonicalized?
        
        Returns:
            canonicalized expression, constraints
        """
        if type(expr) in self.canon_methods:
            return self.canon_methods[type(expr)](expr, args)
        else:
            return expr.copy(args), []

    def invert(self, solution, inverse_data):
        solution = super(Dgp2Dcp, self).invert(solution, inverse_data)
        if solution.status == settings.SOLVER_ERROR:
            return solution
        for vid, value in solution.primal_vars.items():
            solution.primal_vars[vid] = np.exp(value)
        # f(x) = e^{F(u)}.
        solution.opt_val = np.exp(solution.opt_val)
        return solution
