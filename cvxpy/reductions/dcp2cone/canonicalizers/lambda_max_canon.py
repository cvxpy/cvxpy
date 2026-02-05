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

from cvxpy.atoms.affine.diag import diag_vec
from cvxpy.atoms.affine.promote import promote
from cvxpy.atoms.affine.upper_tri import upper_tri
from cvxpy.constraints.psd import PSD
from cvxpy.expressions.variable import Variable
from cvxpy.utilities.bounds import get_expr_bounds_if_supported
from cvxpy.utilities.solver_context import SolverInfo
from cvxpy.utilities.values import get_expr_value_if_supported


def lambda_max_canon(expr, args, solver_context: SolverInfo | None = None):
    A = args[0]
    n = A.shape[0]
    bounds = get_expr_bounds_if_supported(expr, solver_context)
    t = Variable(bounds=bounds)
    value = get_expr_value_if_supported(expr, solver_context)
    if value is not None:
        t.value = value
    prom_t = promote(t, (n,))
    # Constrain I*t - A to be PSD; note that this expression must be symmetric.
    tmp_expr = diag_vec(prom_t) - A
    constr = [PSD(tmp_expr)]
    if not A.is_symmetric():
        ut = upper_tri(A)
        lt = upper_tri(A.T)
        constr.append(ut == lt)
    return t, constr
