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
    print "difference in solution:", sum((cvxopt_x - cvxpy_x)**2)
    print "difference in objective:", abs(obj - status)
else:
    print "Generated infeasible problem"
    print "  ", status
