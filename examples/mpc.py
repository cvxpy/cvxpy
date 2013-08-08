from cvxpy import *
import cvxopt
# Problem data
T = 10
n,p = (10,5)
A = cvxopt.normal(n,n)
B = cvxopt.normal(n,p)
x_init = cvxopt.normal(n)
x_final = cvxopt.normal(n)

# Recursively define MPC.
class Stage(object):
    def __init__(self, past=None):
        self.past = past
        if past is None: # If no past, use default initialization.
            x,cost,constraints = (x_init,0,[]) 
        else:
            x,cost,constraints = (past.x,past.cost,past.constraints)
        self.x = Variable(n)
        self.u = Variable(p)
        self.cost = sum(square(self.u)) + cost
        self.constraints = [self.x == A*x + B*self.u] + constraints

s = Stage()
for i in range(T):
    s = Stage(s)

print Problem(Minimize(s.cost), s.constraints + [s.x == x_final]).solve()

# Iteratively define MPC with shared variables.
class StageIter(object):
    def __init__(self, x):
        self.x = Variable(n)
        self.u = Variable(p)
        self.cost = sum(square(self.u))
        self.constraint = (self.x == A*x + B*self.u)

stages = [StageIter(x_init)] 
for i in range(T):
    stages.append(StageIter(stages[-1].x))

obj = sum(s.cost for s in stages)
constraints = [stages[-1].x == x_final]
map(constraints.append, (s.constraint for s in stages))
print Problem(Minimize(obj), constraints).solve()

# MPC with waypoints and rendezvous.
class StageTarget(object):
    def __init__(self, x, target=None):
        self.x = Variable(n)
        self.u = Variable(p)
        self.cost = sum(square(self.u))
        self.constraints = [self.x == A*x + B*self.u]
        if target is not None:
            self.constraints += [self.x == target]

targets = {T: x_final, 3: -x_final}
path1 = [StageTarget(x_init)]
for i in range(T):
    path1.append(StageTarget(path1[-1].x, targets.get(i+1, None)))

targets = {T: 2*x_final, 3: stages[3].x}
path2 = [StageTarget(2*x_init)]
for i in range(T):
    path2.append(StageTarget(path2[-1].x, targets.get(i+1, None)))

obj = sum(p1.cost + p2.cost for p1,p2 in zip(path1,path2))
constraints = []
map(constraints.extend, (p1.constraints + p2.constraints for p1,p2 in zip(path1,path2)))
print Problem(Minimize(obj), constraints).solve()