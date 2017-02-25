import cvxpy as cvx
import numpy.random as r
import numpy as np
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.settings import ECOS
from cvxpy.tests.test_constant_atoms import atoms, run_atom, test_atom


n = 5
x = cvx.Variable(n)
y = cvx.Variable()
A = r.rand(n,n)
b = r.rand(n,1)

l = np.random.randn(5, 4)
c = [cvx.abs(A*x + b) <= 2, cvx.abs(y) + x[0] <= 1, cvx.log1p(x) >= 5]
c.append(cvx.log_sum_exp(l, axis=0) <= 10)
X = cvx.Variable(5, 5)
c.append(cvx.log_det(X) >= 10)
cvx.Minimize(x[0])
prob = cvx.Problem(cvx.Minimize(x[0] + x[1] + y), c)
 
d2c = Dcp2Cone()
d2c.accepts(prob)
new_prob = d2c.apply(prob)

print(prob)
print('\n\n')
print(new_prob)

prob.solve()
new_prob.solve()

print(prob.value)
print(new_prob.value)
