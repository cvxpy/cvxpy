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

from noncvx_variable import NonCvxVariable
import cvxpy

# Use ADMM to attempt non-convex problem.
def admm(self, rho=0.5, iterations=5, solver=cvxpy.ECOS):
    objective,eq_constr,ineq_constr,dims = self.canonicalize()
    variables = self.variables(objective, eq_constr + ineq_constr)
    noncvx_vars = [obj for obj in variables if isinstance(obj, NonCvxVariable)]
    # Form ADMM problem.
    obj = self.objective.expr
    for var in noncvx_vars:
        obj = obj + (rho/2)*sum(cvxpy.square(var - var.z + var.u))
    p = cvxpy.Problem(cvxpy.Minimize(obj), self.constraints)
    # ADMM loop
    for i in range(iterations):
        p.solve(solver=solver)
        for var in noncvx_vars:
            var.z.value = var.round(var.value + var.u.value)
            var.u.value = var.value - var.z.value
    # Fix noncvx variables and solve.
    fix_constr = []
    for var in noncvx_vars:
        fix_constr += var.fix(var.z.value)
    p = cvxpy.Problem(self.objective, self.constraints + fix_constr)
    return p.solve(solver=solver)

# Add admm method to cvxpy Problem.
cvxpy.Problem.register_solve("admm", admm)