from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import MulExpression, DivExpression, multiply
from cvxpy.atoms.affine.sum import Sum
from cvxpy.atoms.affine.trace import trace
from cvxpy.atoms.elementwise.exp import exp
from cvxpy.atoms.elementwise.log import log
from cvxpy.atoms.elementwise.power import power
from cvxpy.atoms.elementwise.maximum import maximum
from cvxpy.atoms.elementwise.minimum import minimum
from cvxpy.atoms.geo_mean import geo_mean
from cvxpy.atoms.gmatmul import gmatmul
from cvxpy.atoms.one_minus_pos import one_minus_pos
from cvxpy.atoms.eye_minus_inv import eye_minus_inv
from cvxpy.atoms.max import max
from cvxpy.atoms.min import min
from cvxpy.atoms.norm1 import norm1
from cvxpy.atoms.norm_inf import norm_inf
from cvxpy.atoms.pf_eigenvalue import pf_eigenvalue
from cvxpy.atoms.pnorm import pnorm
from cvxpy.atoms.prod import Prod
from cvxpy.atoms.quad_form import quad_form
from cvxpy.atoms.quad_over_lin import quad_over_lin
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.eliminate_pwl.atom_canonicalizers import CANON_METHODS as PWL_METHODS
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.add_canon import add_canon
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.constant_canon import constant_canon
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.div_canon import div_canon
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.exp_canon import exp_canon
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.eye_minus_inv_canon import eye_minus_inv_canon
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.geo_mean_canon import geo_mean_canon
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.gmatmul_canon import gmatmul_canon
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.log_canon import log_canon
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.mul_canon import mul_canon
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.mulexpression_canon import mulexpression_canon
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.norm1_canon import norm1_canon
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.norm_inf_canon import norm_inf_canon
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.one_minus_pos_canon import one_minus_pos_canon
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.pf_eigenvalue_canon import pf_eigenvalue_canon
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.pnorm_canon import pnorm_canon
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.power_canon import power_canon
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.prod_canon import prod_canon
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.quad_form_canon import quad_form_canon
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.quad_over_lin_canon import quad_over_lin_canon
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.sum_canon import sum_canon
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.trace_canon import trace_canon

import numpy as np


CANON_METHODS = {
    AddExpression : add_canon,
    Constant : constant_canon,
    DivExpression : div_canon,
    exp : exp_canon,
    eye_minus_inv : eye_minus_inv_canon,
    geo_mean : geo_mean_canon,
    gmatmul : gmatmul_canon,
    log : log_canon,
    MulExpression : mulexpression_canon,
    multiply : mul_canon,
    norm1 : norm1_canon,
    norm_inf : norm_inf_canon,
    one_minus_pos : one_minus_pos_canon,
    pf_eigenvalue : pf_eigenvalue_canon,
    pnorm : pnorm_canon,
    power : power_canon, 
    Prod : prod_canon,
    quad_form : quad_form_canon,
    quad_over_lin : quad_over_lin_canon,
    trace : trace_canon,
    Sum : sum_canon,
    Variable : None,
    Parameter : None,
}

CANON_METHODS[max] = PWL_METHODS[max]
CANON_METHODS[min] = PWL_METHODS[min]
CANON_METHODS[maximum] = PWL_METHODS[maximum]
CANON_METHODS[minimum] = PWL_METHODS[minimum]

# Canonicalization of DGPs is a stateful procedure, hence the need for a class.
class DgpCanonMethods(dict):
    def __init__(self, *args, **kwargs) -> None:
        super(DgpCanonMethods, self).__init__(*args, **kwargs)
        self._variables = {}
        self._parameters = {}

    def __contains__(self, key):
        return key in CANON_METHODS

    def __getitem__(self, key):
        if key == Variable:
            return self.variable_canon
        elif key == Parameter:
            return self.parameter_canon
        else:
            return CANON_METHODS[key]

    def variable_canon(self, variable, args):
        del args
        # Swaps out positive variables for unconstrained variables.
        if variable in self._variables:
            return self._variables[variable], []
        else:
            log_variable =  Variable(variable.shape, var_id=variable.id)
            self._variables[variable] = log_variable
            return log_variable, []

    def parameter_canon(self, parameter, args):
        del args
        # Swaps out positive parameters for unconstrained variables.
        if parameter in self._parameters:
            return self._parameters[parameter], []
        else:
            log_parameter = Parameter(parameter.shape, name=parameter.name(),
                                      value=np.log(parameter.value))
            self._parameters[parameter] = log_parameter
            return log_parameter, []
