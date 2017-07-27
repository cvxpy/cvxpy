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
    obj = self.objective.args[0]
    for var in noncvx_vars:
        obj = obj + (rho/2)*cvx.sum_entries(cvx.square(var - var.z + var.u))
    prob = cvx.Problem(cvx.Minimize(obj), self.constraints)
    # ADMM loop
    for i in range(iterations):
        result = prob.solve(*args, **kwargs)
        print "relaxation", result
        for idx, var in enumerate(noncvx_vars):
            var.z.value = var.round(var.value + var.u.value)
            # print idx, var.z.value, var.value, var.u.value
            var.u.value += var.value - var.z.value
    return polish(self, noncvx_vars, *args, **kwargs)

# Use ADMM to attempt non-convex problem.
def admm2(self, rho=0.5, iterations=5, *args, **kwargs):
    noncvx_vars = []
    for var in self.variables():
        if getattr(var, "noncvx", False):
            noncvx_vars += [var]
    # Form ADMM problem.
    obj = self.objective.args[0]
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
