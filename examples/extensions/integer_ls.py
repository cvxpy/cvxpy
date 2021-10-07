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

import cvxopt
import ncvx.branch_and_bound
from ncvx.boolean import Boolean

from cvxpy import Minimize, Problem, sum_squares

x = Boolean(3, name='x')
A = cvxopt.matrix([1,2,3,4,5,6,7,8,9], (3, 3), tc='d')
z = cvxopt.matrix([3, 7, 9])

p = Problem(Minimize(sum_squares(A*x - z))).solve(method="branch and bound")

print(x.value)
print(p)

# even a simple problem like this introduces too many variables
# y = Boolean()
# Problem(Minimize(square(y - 0.5))).branch_and_bound()
