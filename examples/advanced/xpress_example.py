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
# s.t. ||y - y0||^2 <= 0.01
#      ||y + y0||^2 <= 0.01
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

result = prob.solve (solver = 'XPRESS', scaling = 0, bargaptarget = 4e-30, original_problem = prob)

# We can now gather data from the Xpress problem, much in the same way
# as in CVXPY, but we now have the Xpress problem object:

data = prob.get_problem_data (solver = 'XPRESS')

p = data['XPRESSprob']

# This problem object can be saved in LP and MPS format (first and
# second instructions. Note that the MPS format is preferable when
# interacting with us as it guarantees that all data is retained. The
# next instruction shows how to use the Xpress problem object to
# retrieve, for instance, two of its attributes.

p.write ('xprob', 'lp')
p.write ('xprob', '')

print ("Problem has {0:4d} columns and {1:4d} rows".format (p.attributes.cols, p.attributes.rows))

# Obtaining the problem's IISs is done through the XpressProblem
# object, rather than p, as we agreed. For LP problems there can be
# more than one IISs, while for quadratic problem only one IIS is
# returned. They are all stored in the field prob._iis of the
# XpressProblem object. The field prob._iis is a list of dictionaries,
# each dictionary corresponding to an IIS. In the following, they are
# all printed if the problem is infeasible, otherwise the variable
# values and the dual value of the first constraint are printed.

print (prob._transferRow)

if prob.status == 'infeasible':
    print (prob._iis)
else:
    # The optimal value for x is stored in x.value.
    print (x.value)
    print (y.value)

    # The optimal Lagrange multiplier for a constraint
    # is stored in constraint.dual_value.
    print (constraints[0].dual_value)

# Since the problem is infeasible and quadratic, the IIS printed is as follows:
#
# [{'origrow' : [blah, quad_con2],
#   'row': [lc_3_0, linT_qc1_0, linT_qc1_3, cone_qc1,
#           lc_5_0, linT_qc2_0, linT_qc2_3, cone_qc2],
#   'rtype': ['L', 'L', 'L', 'L', 'L', 'L', 'G', 'L'],
#   'isolrow': ['-1', '-1', '-1', '-1', '-1', '-1', '-1', '-1'],
#   'col': [cX1_0, cX2_0],
#   'btype': ['L', 'L'],
#   'isolcol': ['-1', '-1'],
#   'duals': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#   'redcost': [0.0, 0.0]}]
#
# the important entries of this dictionary are those for keys 'row'
# (marking what rows are responsible for infeasibility) and 'col'
# (whose bounds are also responsible for the infeasibility of the
# problem). In order to give a meaning to these names, the p.write
# instruction above resulted in the LP file below.
#
# For clarity, constraints have different names: lc_X_Y correspond to
# the original problem's constraints, where X is the index for the
# original constraint and Y is a sub-index in case of vector
# constraint. Hence lc_3_0 correspond to "sum_squares (y - y0) <=
# 0.01", while lc_7_0 and lc_7_1 correspond to the vector constraint x
# >= x0.
#
# Constraints without a direct link to the original problem (because
# they are conic transformations of a quadratic constraint or the
# quadratic objective) are marked as linT (for linear transformation)
# and cone_qcX for the cone constraint.
#
# A similar reasoning applies to variables: while x_* are original
# variables, aux* are variables that have been introduced due to the
# extra (i.e., non-original) constraints. The MPS file has the same
# names as the LP file, but has a much less readable format.

# \Problem name: CVXproblem
#
# Minimize
#  x_1_0 + x_1_1 + aux_4
#
# Subject To
# sumxy_limited_0: aux_5 <= 0.01
# lc_5_0: aux_6 <= 0.01
# lc_7_0: - x_0_0 <= -0
# lc_7_1: - x_0_1 <= -1
# lc_9_0: x_0_0 <= 10
# lc_9_1: x_0_1 <= 20
# linT_qc0_0: - aux_4 + cX0_0 = 1
# linT_qc0_1: aux_4 + cX0_1 = 1
# linT_qc0_2: - 2 x_0_0 + cX0_2 = -0
# linT_qc0_3: - 2 x_0_1 + cX0_3 = -0
# cone_qc0: [ - cX0_0^2 + cX0_1^2 + cX0_2^2 + cX0_3^2 ] <= -0
# linT_qc1_0: - aux_5 + cX1_0 = 1
# linT_qc1_1: aux_5 + cX1_1 = 1
# linT_qc1_2: - 2 x_1_0 + cX1_2 = -0
# linT_qc1_3: - 2 x_1_1 + cX1_3 = -2
# cone_qc1: [ - cX1_0^2 + cX1_1^2 + cX1_2^2 + cX1_3^2 ] <= -0
# linT_qc2_0: - aux_6 + cX2_0 = 1
# linT_qc2_1: aux_6 + cX2_1 = 1
# linT_qc2_2: - 2 x_1_0 + cX2_2 = -0
# linT_qc2_3: - 2 x_1_1 + cX2_3 = 2
# cone_qc2: [ - cX2_0^2 + cX2_1^2 + cX2_2^2 + cX2_3^2 ] <= -0
#
# Bounds
# x_0_0 free
# x_0_1 free
# x_1_0 free
# x_1_1 free
# aux_4 free
# aux_5 free
# aux_6 free
# cX0_1 free
# cX0_2 free
# cX0_3 free
# cX1_1 free
# cX1_2 free
# cX1_3 free


# cX2_1 free
# cX2_2 free
# cX2_3 free
#
# End
