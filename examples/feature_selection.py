from cvxpy import *
from mixed_integer import *
import cvxopt

# Feature selection on a linear kernel SVM classifier.
# Uses the Alternating Direction Method of Multipliers
# with a (non-convex) cardinality constraint.

# Generate data.
cvxopt.setseed(1)
N = 50
M = 40
n = 10
data = []
for i in range(N):
    data += [(1,cvxopt.normal(n, mean=1.0, std=2.0))]
for i in range(M):
    data += [(-1,cvxopt.normal(n, mean=-1.0, std=2.0))]

# Construct problem.
gamma = Parameter(sign="positive")
gamma.value = 0.1
# 'a' is a variable constrained to have at most 6 non-zero entries.
a = SparseVar(n,nonzeros=6)
b = Variable()

slack = [pos(1 - label*(sample.T*a - b)) for (label,sample) in data]
objective = Minimize(norm2(a) + gamma*sum(slack))
p = Problem(objective)
# Extensions can attach new solve methods to the CVXPY Problem class. 
p.solve(method="admm")

# Count misclassifications.
error = 0
for label,sample in data:
    if not label*(a.value.T*sample - b.value)[0] >= 0:
        error += 1

print "%s misclassifications" % error
print a.value
print b.value