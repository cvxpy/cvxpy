import cvxpy as cp

#x = cp.Parameter()
#y = cp.Variable()
#z = x * cp.exp(y)
#print(z.is_dpp())
#print(z.is_dcp())
#
#
#x = cp.Parameter(pos=True)
#y = cp.Variable()
#z = x * cp.exp(y)
#print(z.is_dpp())
#print(z.is_dcp())

x = cp.Parameter(pos=True)
y = cp.Variable(pos=True)
z = cp.exp(x) * y
print(z.is_dpp())
print(z.is_dcp())
