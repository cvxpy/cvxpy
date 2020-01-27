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

from cvxpy import Variable, Problem, Minimize, log
import cvxopt

cvxopt.solvers.options['show_progress'] = False

# create problem data
m, n = 5, 10
A = cvxopt.normal(m,n)
tmp = cvxopt.uniform(n,1)
b = A*tmp

x = Variable(n)

p = Problem(
    Minimize(-sum(log(x))),
    [A*x == b]
)
status = p.solve()
cvxpy_x = x.value

def acent(A, b):
    m, n = A.size
    def F(x=None, z=None):
        if x is None: return 0, cvxopt.matrix(1.0, (n,1))
        if min(x) <= 0.0: return None
        f = -sum(cvxopt.log(x))
        Df = -(x**-1).T
        if z is None: return f, Df
        H = cvxopt.spdiag(z[0] * x**-2)
        return f, Df, H
    sol = cvxopt.solvers.cp(F, A=A, b=b)
    return sol['x'], sol['primal objective']

x, obj = acent(A,b)
cvxopt_x = x

if isinstance(status, (float, int)):
    print("difference in solution:", sum((cvxopt_x - cvxpy_x)**2))
    print("difference in objective:", abs(obj - status))
else:
    print("Generated infeasible problem")
    print("  ", status)
