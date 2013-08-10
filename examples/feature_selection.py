from cvxpy import *
from mixed_integer import *
import cvxopt

# SVM with feature selection using cardinality constraints.
# Generate data.
cvxopt.setseed(2)
N = 50
M = 40
n = 10
x = [cvxopt.normal(n, mean=1.0, std=2.0) for i in range(N)]
y = [cvxopt.normal(n, mean=-1.0, std=2.0) for i in range(M)]

# Construct problem.
gamma = 0.1
a = Variable(n)
b = Variable()
u = Variable(N)
v = Variable(M)

obj = Minimize(norm2(a) + gamma*(sum(u) + sum(v)))
constraints = [u >= 0, v >= 0]
for i in range(N):
    constraints += [x[i].T*a - b >= 1 - u[i]]
for i in range(M):
    constraints += [y[i].T*a - b <= -(1 - v[i])]

p = Problem(obj, constraints + [max_card(a,6)])
p.admm()

# Count misclassifications.
error = 0
for xi in x:
    if not (a.value.T*xi - b.value)[0] >= 0:
        error = error + 1
for yi in y:
    if not (a.value.T*yi - b.value)[0] <= 0:
        error = error + 1

print "%s misclassifications" % error
print a.value
print b.value