"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from cvxpy.atoms.dist_ratio import dist_ratio
from cvxpy.atoms.eye_minus_inv import eye_minus_inv, resolvent
from cvxpy.atoms.geo_mean import geo_mean
from cvxpy.atoms.gen_lambda_max import gen_lambda_max
from cvxpy.atoms.harmonic_mean import harmonic_mean
from cvxpy.atoms.lambda_max import lambda_max
from cvxpy.atoms.lambda_min import lambda_min
from cvxpy.atoms.lambda_sum_largest import lambda_sum_largest
from cvxpy.atoms.lambda_sum_smallest import lambda_sum_smallest
from cvxpy.atoms.length import length
from cvxpy.atoms.log_det import log_det
from cvxpy.atoms.log_sum_exp import log_sum_exp
from cvxpy.atoms.matrix_frac import matrix_frac
from cvxpy.atoms.matrix_frac import MatrixFrac
from cvxpy.atoms.max import max
from cvxpy.atoms.min import min
from cvxpy.atoms.norm import norm, norm2
from cvxpy.atoms.norm1 import norm1
from cvxpy.atoms.norm_inf import norm_inf
from cvxpy.atoms.norm_nuc import normNuc
from cvxpy.atoms.mixed_norm import mixed_norm
from cvxpy.atoms.norm1 import norm1
from cvxpy.atoms.norm_inf import norm_inf
from cvxpy.atoms.one_minus_pos import one_minus_pos
from cvxpy.atoms.one_minus_pos import diff_pos
from cvxpy.atoms.pf_eigenvalue import pf_eigenvalue
from cvxpy.atoms.pnorm import pnorm, Pnorm
from cvxpy.atoms.prod import prod, Prod
from cvxpy.atoms.quad_form import quad_form, QuadForm
from cvxpy.atoms.quad_over_lin import quad_over_lin
from cvxpy.atoms.sigma_max import sigma_max
from cvxpy.atoms.sign import sign
from cvxpy.atoms.sum_largest import sum_largest
from cvxpy.atoms.sum_smallest import sum_smallest
from cvxpy.atoms.sum_squares import sum_squares
from cvxpy.atoms.total_variation import tv

from cvxpy.atoms.affine.bmat import bmat
from cvxpy.atoms.affine.conj import conj
from cvxpy.atoms.affine.conv import conv
from cvxpy.atoms.affine.cumsum import cumsum
from cvxpy.atoms.affine.diag import diag
from cvxpy.atoms.affine.diff import diff
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.imag import imag
from cvxpy.atoms.affine.kron import kron
from cvxpy.atoms.affine.binary_operators import multiply, matmul
from cvxpy.atoms.affine.promote import promote
from cvxpy.atoms.affine.real import real
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.sum import sum
from cvxpy.atoms.affine.trace import trace
from cvxpy.atoms.affine.upper_tri import upper_tri
from cvxpy.atoms.affine.vec import vec
from cvxpy.atoms.affine.vstack import vstack

from cvxpy.atoms.elementwise.abs import abs
from cvxpy.atoms.elementwise.ceil import ceil, floor
from cvxpy.atoms.elementwise.entr import entr
from cvxpy.atoms.elementwise.exp import exp
from cvxpy.atoms.elementwise.huber import huber
from cvxpy.atoms.elementwise.inv_pos import inv_pos
from cvxpy.atoms.elementwise.kl_div import kl_div
from cvxpy.atoms.elementwise.log import log
from cvxpy.atoms.elementwise.log1p import log1p
from cvxpy.atoms.elementwise.logistic import logistic
from cvxpy.atoms.elementwise.maximum import maximum
from cvxpy.atoms.elementwise.minimum import minimum
from cvxpy.atoms.elementwise.neg import neg
from cvxpy.atoms.elementwise.pos import pos
from cvxpy.atoms.elementwise.power import power
from cvxpy.atoms.elementwise.scalene import scalene
from cvxpy.atoms.elementwise.sqrt import sqrt
from cvxpy.atoms.elementwise.square import square

# TODO(akshayka): Perhaps couple this information with the atom classes
# themselves.
SOC_ATOMS = [
    geo_mean,
    pnorm,
    QuadForm,
    quad_over_lin,
    power,
]

EXP_ATOMS = [
    log_sum_exp,
    log_det,
    entr,
    exp,
    kl_div,
    log,
    log1p,
    logistic,
]

PSD_ATOMS = [
    lambda_max,
    lambda_sum_largest,
    log_det,
    MatrixFrac,
    normNuc,
    sigma_max,
]
