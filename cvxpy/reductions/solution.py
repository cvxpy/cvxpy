# Status codes.
OPTIMAL = "OPTIMAL"
INFEASIBLE = "INFEASIBLE"
UNBOUNDED = "UNBOUNDED"
USER_LIMIT = "USER_LIMIT"
ERROR = "ERROR"

class Solution(object):
    """A solution object.

    Attributes:
        status: status code
        opt_val: float
        primal_vars: dict of id to NumPy ndarray
        dual_vars: dict of id to NumPy ndarray
        attr: dict of other attributes.
    """
    def __init__(self, status=None, opt_val=None, primal_vars=None, dual_vars=None, attr=None):
        self.status = status
        self.opt_val = opt_val
        self.primal_vars = primal_vars
        self.dual_vars = dual_vars
        self.attr = attr
