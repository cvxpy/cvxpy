"""
Copyright 2025 CVXPY developers

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
from cvxpy.atoms.quad_over_lin import quad_over_lin
from cvxpy.atoms.elementwise.exp import exp
from cvxpy.atoms.elementwise.log import log
from cvxpy.atoms.elementwise.entr import entr
from cvxpy.atoms.elementwise.rel_entr import rel_entr
from cvxpy.atoms.elementwise.kl_div import kl_div
from cvxpy.atoms.elementwise.minimum import minimum
from cvxpy.atoms.elementwise.maximum import maximum
from cvxpy.atoms.elementwise.power import power
from cvxpy.atoms.elementwise.trig import cos, sin, tan
from cvxpy.atoms.norm1 import norm1
from cvxpy.atoms.pnorm import Pnorm
from cvxpy.atoms.elementwise.abs import abs
from cvxpy.atoms.affine.binary_operators import DivExpression, MulExpression, multiply
from cvxpy.reductions.dnlp2smooth.canonicalizers.quad_over_lin_canon import quad_over_lin_canon
from cvxpy.reductions.dnlp2smooth.canonicalizers.div_canon import div_canon
from cvxpy.reductions.dnlp2smooth.canonicalizers.log_canon import log_canon
from cvxpy.reductions.dnlp2smooth.canonicalizers.exp_canon import exp_canon
from cvxpy.reductions.dnlp2smooth.canonicalizers.multiply_canon import matmul_canon, multiply_canon
from cvxpy.reductions.dnlp2smooth.canonicalizers.pnorm_canon import pnorm_canon
from cvxpy.reductions.dnlp2smooth.canonicalizers.power_canon import power_canon
from cvxpy.reductions.dnlp2smooth.canonicalizers.entr_canon import entr_canon
from cvxpy.reductions.dnlp2smooth.canonicalizers.rel_entr_canon import rel_entr_canon
from cvxpy.reductions.dnlp2smooth.canonicalizers.kl_div_canon import kl_div_canon
from cvxpy.reductions.dnlp2smooth.canonicalizers.trig_canon import cos_canon, sin_canon, tan_canon
from cvxpy.reductions.eliminate_pwl.canonicalizers.norm1_canon import norm1_canon
from cvxpy.reductions.eliminate_pwl.canonicalizers.max_canon import max_canon
from cvxpy.reductions.eliminate_pwl.canonicalizers.min_canon import min_canon
from cvxpy.reductions.eliminate_pwl.canonicalizers.maximum_canon import maximum_canon
from cvxpy.reductions.eliminate_pwl.canonicalizers.minimum_canon import minimum_canon
from cvxpy.reductions.eliminate_pwl.canonicalizers.abs_canon import abs_canon


SMOOTH_CANON_METHODS = {
    log: log_canon,
    exp: exp_canon,
    sin: sin_canon,
    cos: cos_canon,
    tan: tan_canon,
    quad_over_lin: quad_over_lin_canon,
    power: power_canon,
    Pnorm : pnorm_canon,
    DivExpression: div_canon,
    entr: entr_canon,
    rel_entr: rel_entr_canon,
    kl_div: kl_div_canon,
    multiply: multiply_canon,
    MulExpression: matmul_canon,

    # ESR atoms
    abs: abs_canon,
    maximum: maximum_canon,
    max: max_canon,
    norm1: norm1_canon,

    # HSR atoms
    minimum: minimum_canon,
    min: min_canon,
}
