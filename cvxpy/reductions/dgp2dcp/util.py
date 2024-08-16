"""
Copyright 2024 the CVXPY developers
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

from cvxpy.atoms.affine.vec import vec


# the Python `sum` function is a reduction with initial value 0.0,
# resulting in a non-DGP expression
def explicit_sum(expr):
    x = vec(expr, order='F')
    summation = x[0]
    for xi in x[1:]:
        summation += xi
    return summation
