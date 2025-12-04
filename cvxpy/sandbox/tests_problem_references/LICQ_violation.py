
import cvxpy as cp

# example 1
x1 = cp.Variable(name="x1", nonneg=True)
prob  = cp.Problem(cp.Minimize(cp.sqrt(x1)))
prob.solve(solver=cp.IPOPT, nlp=True, derivative_test='none', verbose=True)
