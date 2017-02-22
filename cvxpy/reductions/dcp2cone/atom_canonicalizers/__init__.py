from cvxpy.atoms import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.abs_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.affine_prod import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.exp_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.entr_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.geo_mean_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.huber_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.kl_div_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.lambda_max_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.log_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.log_det import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.log_sum_exp_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.log1p_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.logistic_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.matrix_frac_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.max_elemwise_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.max_entries_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.normNuc_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.power import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.pnorm import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.sigma_max_canon import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.sum_largest import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers.quad_over_lin_canon import *

CANON_METHODS = {
		affine_prod : affine_prod_canon,
		geo_mean : geo_mean_canon,
		lambda_max : lambda_max_canon,
		log_det : log_det_canon,
		log_sum_exp : log_sum_exp_canon,
		matrix_frac : matrix_frac_canon,
		max_entries : max_entries_canon,
		normNuc : normNuc_canon,
		pnorm : pnorm_canon,
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
		max_elemwise : max_elemwise_canon,
		power : power_canon,
}
