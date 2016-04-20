__author__ = 'Xinyue'

import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.lin_ops.lin_op as lo
from cvxpy.problems.objective import Minimize, Maximize
from cvxpy.problems.problem import Problem
from cvxpy.expressions.variables import Variable
from cvxpy.expressions.expression import Expression
import copy
import numpy as np

def linearize(expr):
    line_expr = expr.value
    for key in expr.gradient:
        line_expr += expr.gradient[key].T * (key - key.value)
    return line_expr
