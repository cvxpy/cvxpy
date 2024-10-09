import cvxpy as cp

x = cp.Variable(2)
prob = cp.Problem(cp.Minimize(cp.inv_prod(x[:1])+cp.inv_prod(x[:2])), [cp.sum(x)==2])

print(x[0])
prob.solve()
print(x.value)
print(prob.value) # Wrong

x = cp.Variable(2)
prob = cp.Problem(cp.Minimize(cp.inv_pos(x[0])+cp.inv_prod(x[:2])), [cp.sum(x)==2])
prob.solve()
print(x.value)
print(prob.value) # Correct
