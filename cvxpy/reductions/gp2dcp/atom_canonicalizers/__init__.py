from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import multiply, DivExpression
from cvxpy.atoms.elementwise.log import log
from cvxpy.atoms.elementwise.power import power
from cvxpy.atoms.elementwise.maximum import maximum
from cvxpy.atoms.sum_largest import sum_largest
from cvxpy.constraints.nonpos import NonPos
from cvxpy.constraints.zero import Zero
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.eliminate_pwl.atom_canonicalizers.maximum_canon import maximum_canon
from cvxpy.reductions.eliminate_pwl.atom_canonicalizers.sum_largest_canon import sum_largest_canon
from cvxpy.reductions.gp2dcp.atom_canonicalizers.add_canon import add_canon
from cvxpy.reductions.gp2dcp.atom_canonicalizers.constant_canon import constant_canon
from cvxpy.reductions.gp2dcp.atom_canonicalizers.div_canon import div_canon
from cvxpy.reductions.gp2dcp.atom_canonicalizers.zero_constr_canon import zero_constr_canon
from cvxpy.reductions.gp2dcp.atom_canonicalizers.nonpos_constr_canon import nonpos_constr_canon
from cvxpy.reductions.gp2dcp.atom_canonicalizers.mul_canon import mul_canon
from cvxpy.reductions.gp2dcp.atom_canonicalizers.power_canon import power_canon


# TODO(akshayka): canon for matrix multiplication (MulExpression)
CANON_METHODS = {
    AddExpression : add_canon,
    Constant : constant_canon,
    multiply : mul_canon,
    power : power_canon, 
    Variable : None,
    Zero : zero_constr_canon,
    NonPos : nonpos_constr_canon,
    maximum : maximum_canon,
    DivExpression : div_canon,
    sum_largest : sum_largest_canon,  # TODO(akshayka): Figure out sum_largest canon
}

# TODO(akshayka): This is a hack, think of something nicer.
class DgpCanonMethods(dict):
    def __init__(self, *args, **kwargs):
        super(DgpCanonMethods, self).__init__(*args, **kwargs)
        self.update(CANON_METHODS)
        self._variables = {}

    def __getitem__(self, key):
        if key == Variable:
            return self.variable_canon
        else:
            return CANON_METHODS[key]

    def variable_canon(self, expr, args):
        if expr in self._variables:
            return self._variables[expr], []
        else:
            log_variable =  Variable(expr.shape, var_id=expr.id)
            self._variables[expr] = log_variable
            return log_variable, []
