from cvxpy import *
import mixed_integer as ma
import cvxopt

# SVM with feature selection using cardinality constraints.
# Generate data.
N = 50
M = 40
n = 10
x = [cvxopt.normal(n, mean=1.0, std=2.0) for i in range(N)]
y = [cvxopt.normal(n, mean=-1.0, std=2.0) for i in range(M)]

# Construct problem.
gamma = 0.1
a = ma.Variable(n, name='a')
a.max_card(4)
b = Variable(name='b')
u = Variable(N, name='u')
v = Variable(M, name='v')

obj = Minimize(norm2(a) + gamma*(sum(u) + sum(v)))
constraints = [u >= 0, v >= 0]
for i in range(N):
    constraints += [x[i].T*a - b >= 1 - u[i]]
for i in range(M):
    constraints += [y[i].T*a - b <= -(1 - v[i])]

p = ma.Problem(obj, constraints)
p.solve()

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