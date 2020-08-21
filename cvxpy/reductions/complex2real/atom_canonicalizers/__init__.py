"""
Copyright 2017 Robin Verschueren

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

from cvxpy.atoms import (bmat, cumsum, kron, conv,
                         abs, reshape, trace,
                         upper_tri, conj, imag, real,
                         norm1, norm_inf, Pnorm,
                         sigma_max, lambda_max, lambda_sum_largest,
                         log_det, QuadForm, MatrixFrac, quad_over_lin)
from cvxpy.atoms.affine.diag import diag_mat, diag_vec
from cvxpy.atoms.affine.promote import Promote
from cvxpy.atoms.affine.sum import Sum
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.index import index, special_index
from cvxpy.atoms.affine.unary_operators import NegExpression
from cvxpy.atoms.affine.transpose import transpose
from cvxpy.atoms.affine.hstack import Hstack
from cvxpy.atoms.affine.vstack import Vstack
from cvxpy.atoms.norm_nuc import normNuc
from cvxpy.atoms.affine.binary_operators import (MulExpression,
                                                 multiply,
                                                 DivExpression)
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.constraints import Equality, Inequality, NonPos, SOC, PSD, Zero, NonNeg
from cvxpy.reductions.complex2real.atom_canonicalizers.abs_canon import abs_canon
from cvxpy.reductions.complex2real.atom_canonicalizers.aff_canon import (separable_canon,
                                                                         real_canon,
                                                                         imag_canon,
                                                                         conj_canon,
                                                                         binary_canon)
from cvxpy.reductions.complex2real.atom_canonicalizers.pnorm_canon import pnorm_canon
from cvxpy.reductions.complex2real.atom_canonicalizers.matrix_canon import (
    hermitian_canon, quad_canon, lambda_sum_largest_canon, norm_nuc_canon, matrix_frac_canon,
    quad_over_lin_canon)
from cvxpy.reductions.complex2real.atom_canonicalizers.inequality_canon import (
    inequality_canon, nonpos_canon, nonneg_canon)
from cvxpy.reductions.complex2real.atom_canonicalizers.psd_canon import psd_canon
from cvxpy.reductions.complex2real.atom_canonicalizers.soc_canon import soc_canon
from cvxpy.reductions.complex2real.atom_canonicalizers.variable_canon import variable_canon
from cvxpy.reductions.complex2real.atom_canonicalizers.constant_canon import constant_canon
from cvxpy.reductions.complex2real.atom_canonicalizers.param_canon import param_canon
from cvxpy.reductions.complex2real.atom_canonicalizers.equality_canon import (
    equality_canon, zero_canon)


CANON_METHODS = {
    AddExpression: separable_canon,
    bmat: separable_canon,
    cumsum: separable_canon,
    diag_mat: separable_canon,
    diag_vec: separable_canon,
    Hstack: separable_canon,
    index: separable_canon,
    special_index: separable_canon,
    Promote: separable_canon,
    reshape: separable_canon,
    Sum: separable_canon,
    trace: separable_canon,
    transpose: separable_canon,
    NegExpression: separable_canon,
    upper_tri: separable_canon,
    Vstack: separable_canon,

    conv: binary_canon,
    DivExpression: binary_canon,
    kron: binary_canon,
    MulExpression: binary_canon,
    multiply: binary_canon,

    conj: conj_canon,
    imag: imag_canon,
    real: real_canon,
    Variable: variable_canon,
    Constant: constant_canon,
    Parameter: param_canon,
    Inequality: inequality_canon,
    NonPos: inequality_canon,
    NonNeg: inequality_canon,
    PSD: psd_canon,
    SOC: soc_canon,
    Equality: equality_canon,
    Zero: equality_canon,

    abs: abs_canon,
    norm1: pnorm_canon,
    norm_inf: pnorm_canon,
    Pnorm: pnorm_canon,

    lambda_max: hermitian_canon,
    log_det: norm_nuc_canon,
    normNuc: norm_nuc_canon,
    sigma_max: hermitian_canon,
    QuadForm: quad_canon,
    quad_over_lin: quad_over_lin_canon,
    MatrixFrac: matrix_frac_canon,
    lambda_sum_largest: lambda_sum_largest_canon,
}
