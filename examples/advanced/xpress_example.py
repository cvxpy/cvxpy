# The following shows how to use extended features of the Xpress-CVXPY
# interface. These features have the purpose of making available some
# of Xpress' capabilities to the CVXPY user, such as the Xpress
# problem itself, its attributes, and, if the problem is infeasible,
# its Irreducible Infeasible Subsystems (IISs).
#
# Let's begin with the creation of variables and constraints. This is
# more or less the same as with the CVXPY interface and does not
# require any specialization for Xpress.

from cvxpy import *

#
# First (minor) difference: we must import the XpressProblem class
# from xpress_problem.py in the problems/ directory.
#

from cvxpy.problems.xpress_problem import XpressProblem

import numpy

# Problem data.
n = 2
numpy.random.seed(1)
x0 = numpy.arange (n)
y0 = numpy.arange (n)

# Construct the problem.
x = Variable (n)    # x_0_0, x_0_1
y = Variable (n)    # x_1_0, x_1_1

x.var_id = 'Xvar'
y.var_id = 'Y'

objective = Minimize (sum (y) + sum_squares (x))

qcon1 = sum_squares (y - y0) <= 1.01
lowx  = x >= x0
upx   = x <= 10 + 10 * x0
qcon2 = sum_squares (y + y0) <= 1.01

qcon1.constr_id = 'dist_pos'
lowx.constr_id  = 'first_orthant'
upx.constr_id   = 'upper_lim'
qcon2.constr_id = 'dist_neg'

constraints = [qcon1, lowx, qcon2, upx]

# The above variables, constraints, and objectives correspond to the
# problem
#
# min   sum_i y_i + sum_i x_i^2
#
# s.t. ||y - y0||^2 <= 1.01
#      ||y + y0||^2 <= 1.01
#      x >= x0
#      x <= 10 + 10*x0.
#
# Because y0 is the vector (0,1,2,...), the first two constraints
# (distance of y from y0 AND from -y0 must be at most 0.1) are
# incompatible, and the problem is infeasible.
#
# Next is the declaration of the Xpress problem, much the same way as
# in CVXPY but with a different problem class.

prob = XpressProblem (objective, constraints)

# Calling the solve() method of the XpressProblem class requires to
# pass the usual parameters (although solver = 'XPRESS' should
# probably removed); note that the extra parameter "original_problem =
# prob", which gives Xpress a pointer to the CVXPY problem, which is
# in general not passed to any solver. While not mandatory, this will
# help Xpress' solve() function retrieve a faithful correspondence
# between original constraints and Xpress constraints.
#
# Note that all other Xpress controls (scaling, bargaptarget) can be
# passed to solve() without the need for solver_opts. At the end,
# result will contain the value of the objective function of an
# optimal solution, if one exist, or -inf or +inf if the problem is
# unbounded and infeasible, respectively (in the case of a
# minimization problem).

result = prob.solve(solver='XPRESS', scaling=0, bargaptarget=4e-30,
                    original_problem=prob, write_mps='infeas.mps')

# We can now gather data from the Xpress problem, much in the same way
# as in CVXPY, but we now have the Xpress problem object:

data = prob.get_problem_data (solver = 'XPRESS')

if prob.status != 'infeasible':

    # The optimal value for x is stored in x.value.
    print (x.value)
    print (y.value)

    # The optimal Lagrange multiplier for a constraint
    # is stored in constraint.dual_value.
    print (constraints[0].dual_value)
