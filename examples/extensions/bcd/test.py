__author__ = 'Xinyue'

from cvxpy import *
from bcd import bcd
import numpy as np
from fix import fix_prob

alpha = NonNegative(1)
x = NonNegative(1)
y = NonNegative(1)

#prob = Problem(Minimize(alpha), [x +1.5 <= sqrt(x+0.5)*alpha])
#prob = Problem(Minimize(inv_pos(sqrt(y+0.5))*(x+1.5)), [x==y])
#prob = Problem(Minimize(alpha), [square(x)+1 <= log(x+2)*alpha])
#prob = Problem(Minimize(inv_pos(log(y+2))*(square(x)+1)), [x==y])
#prob = Problem(Minimize(inv_pos(x+1)*exp(y)), [x==y])
prob = Problem(Minimize(alpha), [exp(x) <= (x+1)*alpha])

prob.solve(method = 'bcd',solver = 'SCS', rho=1.1)
print "======= solution ======="
for var in prob.variables():
    print var.name(), "=", var.value
print "objective = ", prob.objective.args[0].value

# bisection
print "==== bisection method ===="
upper = Parameter(sign = 'Positive')
lower = Parameter(sign = 'Positive')
prob = Problem(Minimize(0), [exp(x)<=(upper+lower)*(x+1)/float(2)])
upper.value = 1000
lower.value = 0
flag = 1
while lower.value +1e-3 <= upper.value:
    prob.solve()
    if x.value == None:
        lower.value = (upper+lower).value/float(2)
    else:
        upper.value = (upper+lower).value/float(2)
print "upper = ", upper.value
print "lower = ", lower.value