"""
Copyright 2025 CVXPY developers

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
from cvxpy.expressions.variable import Variable
from cvxpy.utilities.solver_context import SolverInfo


def smooth_full_domain_canon(expr, args, solver_context: SolverInfo | None = None):
    if isinstance(args[0], Variable):
        return expr, []
    t = Variable(args[0].shape)
    if args[0].value is not None:
        t.value = args[0].value
    return expr.copy([t]), [t == args[0]]

sinh_canon = smooth_full_domain_canon
tanh_canon = smooth_full_domain_canon
asinh_canon = smooth_full_domain_canon
exp_canon = smooth_full_domain_canon
logistic_canon = smooth_full_domain_canon
sin_canon = smooth_full_domain_canon
cos_canon = smooth_full_domain_canon
prod_canon = smooth_full_domain_canon
