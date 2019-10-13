import cvxpy as cp

x = cp.Variable(name='x')
y = cp.Parameter(name='y')
z = cp.Parameter(name='z')
prod = x * (y * z)
prob = cp.Problem(cp.Minimize(prod))
red = cp.reductions.ReassociateMul(prob)
new_prob = red.reduce()
print(prob.objective.expr.args)
print(new_prob.objective.expr.args)

x = cp.Variable(name='x')
y = cp.Parameter(name='y')
z = cp.Parameter(name='z')
prod = (y * z) * x
prob = cp.Problem(cp.Minimize(prod))
red = cp.reductions.ReassociateMul(prob)
new_prob = red.reduce()
print(prob.objective.expr.args)
print(new_prob.objective.expr.args)
