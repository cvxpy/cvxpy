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
import numpy as np

from cvxpy.expressions.variable import Variable

"""Common DNLP canonicalizers for smooth atoms. There are four
categories of canonicalizers, grouped by the domain type of the atom:

- Full domain: Atoms like exp, sin, cos that have full domain.
  Two variants exist depending on whether the atom's chain rule is
  implemented in the differentiation engine. Eventually, all atoms
  in this category should use the chain rule variant and the non-chain-rule
  variant can be removed.
- Nonnegative domain: Atoms like log and entr that require nonnegative inputs.
- Bounded domain: Atoms like tan and atanh whose domain is an interval.

All canon functions follow the standard signature

    def canon(expr, args) -> (new_expr, constraints)

where args are the (already-canonicalized) arguments of expr.
"""

MIN_INIT_NONNEG_CANON = 1e-3

def smooth_full_domain_canon_non_chain_rule(expr, args):
    """Canonicalize a smooth atom with full domain (and one argument)
       whose chain rule is not implemented in the differentiation engine."""
    if isinstance(args[0], Variable):
        return expr.copy([args[0]]), []
    t = Variable(args[0].shape)
    if args[0].value is not None:
        t.value = args[0].value
    return expr.copy([t]), [t == args[0]]

def smooth_full_domain_canon_chain_rule(expr, args):
    """Canonicalize a smooth atom with full domain (and potentially multiple
       arguments) """
    return expr.copy(args), []

def smooth_nonnegative_dom_canon(expr, args):
    """Canonicalize a smooth atom (with one argument) whose domain is the
       nonnegative reals."""
    t = Variable(args[0].shape, nonneg=True)
    if args[0].value is not None:
        t.value = np.maximum(args[0].value, MIN_INIT_NONNEG_CANON)
    return expr.copy([t]), [t == args[0]]

def make_smooth_range_dom_canon(lower, upper):
    """Wrapper for canonicalizers whose domain is a bounded interval."""
    def smooth_range_dom_canon(expr, args):
        t = Variable(args[0].shape, bounds=[lower, upper])
        # If the initial value is less than 10% of the total interval length from
        # one of the bounds, we initialize at the midpoint of the interval instead.
        margin = 0.1 * (upper - lower)
        midpoint = (lower + upper) / 2.0
        if args[0].value is not None:
            safe_idxs = (args[0].value >= lower + margin) & (args[0].value <= upper - margin)
            trimmed_value = np.where(safe_idxs, args[0].value, midpoint)
            t.value = trimmed_value
        return expr.copy([t]), [t == args[0]]
    return smooth_range_dom_canon

# atoms with domain equal to the nonnegative reals
entr_canon = smooth_nonnegative_dom_canon
log_canon = smooth_nonnegative_dom_canon

# atoms that do not yet have chain rule implemented in diff engine
prod_canon = smooth_full_domain_canon_non_chain_rule

# elementwise atoms with chain rule implemented in diff engine
exp_canon = smooth_full_domain_canon_chain_rule
sin_canon = smooth_full_domain_canon_chain_rule
cos_canon = smooth_full_domain_canon_chain_rule
sinh_canon = smooth_full_domain_canon_chain_rule
tanh_canon = smooth_full_domain_canon_chain_rule
asinh_canon = smooth_full_domain_canon_chain_rule
logistic_canon = smooth_full_domain_canon_chain_rule
normcdf_canon = smooth_full_domain_canon_chain_rule

# other atoms with chain rule implemented in diff engine
multiply_canon = smooth_full_domain_canon_chain_rule
matmul_canon = smooth_full_domain_canon_chain_rule
quad_form_canon = smooth_full_domain_canon_chain_rule

# atoms with domain equal to a bounded interval
atanh_canon = make_smooth_range_dom_canon(-1, 1)
tan_canon = make_smooth_range_dom_canon(-np.pi/2, np.pi/2)
