import numpy
import cvxpy as cvx


b = numpy.arange(10)
X = cvx.Variable(shape=(10, 2))
expr = cvx.pnorm(X, p=2, axis=1) - b
con = [expr <= 0]
obj = cvx.Maximize(cvx.sum(X))
prob = cvx.Problem(obj, con)
result = prob.solve(solver='ECOS')
