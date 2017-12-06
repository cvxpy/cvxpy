"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

from cvxpy.atoms import *
from cvxpy.transforms.indicator import indicator
from cvxpy.reductions.dcp2cone.atom_canonicalizers.cumsum_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.exp_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.entr_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.geo_mean_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.huber_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.indicator_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.kl_div_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.lambda_max_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.lambda_sum_largest_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.log_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.log_det_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.log_sum_exp_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.log1p_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.logistic_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.matrix_frac_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.normNuc_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.power_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.pnorm_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.sigma_max_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.quad_form_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.quad_over_lin_canon import *

from cvxpy.reductions.eliminate_pwl.atom_canonicalizers import (abs_canon,
    maximum_canon, max_canon, norm1_canon,
    norm_inf_canon, sum_largest_canon)

# TODO: remove pwl canonicalize methods, use EliminatePwl reduction instead
CANON_METHODS = {
    cumsum : cumsum_canon,
    geo_mean : geo_mean_canon,
    lambda_max : lambda_max_canon,
    lambda_sum_largest : lambda_sum_largest_canon,
    log_det : log_det_canon,
    log_sum_exp : log_sum_exp_canon,
    MatrixFrac : matrix_frac_canon,
    max : max_canon,
    norm1 : norm1_canon,
    normNuc : normNuc_canon,
    norm_inf : norm_inf_canon,
    Pnorm : pnorm_canon,
    QuadForm : quad_form_canon,
    quad_over_lin : quad_over_lin_canon,
    sigma_max : sigma_max_canon,
    sum_largest : sum_largest_canon,
    abs : abs_canon,
    entr : entr_canon,
    exp : exp_canon,
    huber : huber_canon,
    kl_div : kl_div_canon,
    log : log_canon,
    log1p : log1p_canon,
    logistic : logistic_canon,
    maximum : maximum_canon,
    power : power_canon,
    indicator : indicator_canon,
}
