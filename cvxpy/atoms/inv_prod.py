
from cvxpy.expressions.expression import Expression
from cvxpy.atoms.geo_mean import geo_mean
from cvxpy.atoms.elementwise.power import power
from cvxpy.atoms.elementwise.inv_pos import inv_pos

def inv_prod(value):
    """The reciprocal of a product of the entries of a vector ``x``.
    """
    return power(inv_pos(geo_mean(value)), sum(value.shape))