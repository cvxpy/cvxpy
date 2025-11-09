import cvxpy as cp

# broadcasting one - this works
x = cp.Variable(2, name='x')
logx = cp.log(x)
expr1 = cp.broadcast_to(logx, (4,2))

# broadcasting two - this causes issues
#expr2 = cp.broadcast_to(logx, (2,4))

# broadcasting three - this works
y = cp.Variable((2, 1), name='y')
logy = cp.log(y)
expr3 = cp.broadcast_to(logy, (2,3))

# broadcasting four - this doesn't work
#z = cp.Variable((2, ), name='z')
#logz = cp.log(z)
#expr4 = cp.broadcast_to(logz, (2,3))

# broadcasting five - this works
z = cp.Variable((2, ), name='z')
logz = cp.log(z)
expr5 = cp.broadcast_to(logz, (4,2))

# broadcasting six - this works
z = cp.Variable((1, 2), name='z')
logz = cp.log(z)
expr6 = cp.broadcast_to(logz, (4,2))

# even more general broadcasting - this works
z = cp.Variable((1, 4), name='z')
y = cp.Variable((3, 1), name='y')
expr7 = z - y


