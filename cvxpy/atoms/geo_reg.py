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

from cvxpy.atoms.elementwise.abs import abs
from cvxpy.atoms.elementwise.inv_pos import inv_pos
from cvxpy.atoms.affine.sum import sum

# geo_reg(x, a) = sum_{k=1}^\infty \|x\|_k^k / a^k
def geo_reg(x, a=2.0):
    assert a > 1, "a must be > 1"
    return (a - 1) * sum(inv_pos(1 - abs(x) / a) - 1)
