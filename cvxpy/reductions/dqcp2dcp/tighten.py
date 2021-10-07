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
import numpy as np

from cvxpy.atoms import ceil, floor, length

integer_valued_fns = set([ceil, floor, length])


# Tuples fns such that that t infeasible implies fns[0](t) infeasible
# (or sup of infeasible set), t feasible implies fns[1](t)
# (or inf of infeasible set)
def tighten_fns(expr):
    if type(expr) in integer_valued_fns:
        return (np.ceil, np.floor)
    elif expr.is_nonneg():
        return (lambda t: np.maximum(t, 0), lambda t: t)
    elif expr.is_nonpos():
        return (lambda t: t, lambda t: np.minimum(t, 0))
    else:
        return (lambda t: t, lambda t: t)
