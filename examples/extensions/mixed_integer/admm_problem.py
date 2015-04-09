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
import cvxpy as cvx
from cvxpy import settings as s
import numpy as np

# Use ADMM to attempt non-convex problem.
def admm(self, rho=0.5, iterations=5, *args, **kwargs):
    noncvx_vars = []
    for var in self.variables():
        if getattr(var, "noncvx", False):
            noncvx_vars += [var]
    # Form ADMM problem.
    obj = self.objective._expr
    for var in noncvx_vars:
        obj = obj + (rho/2)*cvx.sum_entries(cvx.square(var - var.z + var.u))
    prob = cvx.Problem(cvx.Minimize(obj), self.constraints)
    # ADMM loop
    for i in range(iterations):
        result = prob.solve(*args, **kwargs)
        for var in noncvx_vars:
            var.z.value = var.round(var.value + var.u.value)
            var.u.value += var.value - var.z.value
    return polish(self, noncvx_vars, *args, **kwargs)

# Use ADMM to attempt non-convex problem.
def admm2(self, rho=0.5, iterations=5, *args, **kwargs):
    noncvx_vars = []
    for var in self.variables():
        if getattr(var, "noncvx", False):
            noncvx_vars += [var]
    # Form ADMM problem.
    obj = self.objective._expr
    for var in noncvx_vars:
        obj = obj + (rho/2)*cvx.sum_entries(cvx.square(var - var.z + var.u))
    prob = cvx.Problem(cvx.Minimize(obj), self.constraints)
    # ADMM loop
    best_so_far = np.inf
    for i in range(iterations):
        result = prob.solve(*args, **kwargs)
        for var in noncvx_vars:
            var.z.value = var.round(var.value + var.u.value)
            var.u.value += var.value - var.z.value
        polished_opt = polish(self, noncvx_vars, *args, **kwargs)
        if polished_opt < best_so_far:
            best_so_far = polished_opt
            print best_so_far
    return best_so_far

def polish(prob, noncvx_vars, *args, **kwargs):
    # Fix noncvx variables and solve.
    fix_constr = []
    for var in noncvx_vars:
        fix_constr += var.fix(var.z.value)
    prob = cvx.Problem(prob.objective, prob.constraints + fix_constr)
    return prob.solve(*args, **kwargs)

# Add admm method to cvx Problem.
cvx.Problem.register_solve("admm", admm)
cvx.Problem.register_solve("admm2", admm2)
