from cvxpy import *
X = Variable((3, 2))
obj = norm(X, 'fro')
prob = Problem(Minimize(obj))
prob.solve()
