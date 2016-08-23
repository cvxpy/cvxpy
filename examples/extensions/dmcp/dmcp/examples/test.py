__author__ = 'Xinyue'

from cvxpy import *
from examples.extensions.dmcp.dmcp.dmcp import bcd


#alpha = NonNegative(1)
#alpha.value = 1
#x = NonNegative(1)
#y = NonNegative(1)

x = Variable(1)
y = Variable(1)
z = Variable(1)
w = Variable(1)

x.value = 2.0
y.value = 1
z.value  = 0.6
w.value = 5.0

#prob = Problem(Minimize(alpha), [square(x) +1 <= sqrt(x+0.5)*alpha])
#prob = Problem(Minimize(inv_pos(sqrt(y+0.5))*(square(x) +1)), [x==y])
#prob = Problem(Minimize(alpha), [square(x)+1 <= log(x+2)*alpha])
#prob = Problem(Minimize(inv_pos(log(y+2))*(square(x)+1)), [x==y])
#prob = Problem(Minimize(inv_pos(x+1)*exp(y)), [x==y])
#prob = Problem(Minimize(alpha), [exp(x) <= (x+1)*alpha])
prob = Problem(Minimize(abs(x*y+z*w)),[x+y+z+w==1])

prob.solve(method = 'bcd', ep = 1e-4, rho = 1.1)
print "======= solution ======="
for var in prob.variables():
    print var.name(), "=", var.value
print "objective = ", prob.objective.args[0].value


# bisection
print "==== bisection method ===="
upper = Parameter(sign = 'Positive')
lower = Parameter(sign = 'Positive')
prob = Problem(Minimize(0), [square(x) +1<=(upper+lower)*sqrt(x+0.5)/float(2)])
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
