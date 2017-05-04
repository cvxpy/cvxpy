#from cvxpy.constraints.semidefinite import SDP
from cvxpy.constraints.psd import PSD
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variables.variable import Variable
import numpy as np


def lambda_max_canon(expr, args):
    A = args[0]
    shape = expr.shape
    t = Variable(*shape)
    # SDP constraint: I*t - A
    # TODO(akshayka): Is there a more efficient way to represent I*t - A?
    expr = Constant(np.eye(A.shape[0])) * t - A
    return t, [PSD(expr)]
