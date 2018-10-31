from cvxpy.expressions.constants.constant import Constant

import numpy as np


def constant_canon(expr, args):
    del args
    return Constant(np.log(expr.value)), []
