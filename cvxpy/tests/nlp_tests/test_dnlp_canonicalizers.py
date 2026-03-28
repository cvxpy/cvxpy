"""
Copyright, the CVXPY authors

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
import pytest

import cvxpy as cp
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dnlp2smooth.canonicalizers.entr_canon import entr_canon
from cvxpy.reductions.dnlp2smooth.canonicalizers.kl_div_canon import kl_div_canon
from cvxpy.reductions.dnlp2smooth.canonicalizers.log_canon import log_canon
from cvxpy.reductions.dnlp2smooth.canonicalizers.power_canon import power_canon
from cvxpy.reductions.dnlp2smooth.canonicalizers.quad_over_lin_canon import (
    quad_over_lin_canon,
)
from cvxpy.reductions.dnlp2smooth.canonicalizers.rel_entr_canon import rel_entr_canon
from cvxpy.reductions.dnlp2smooth.canonicalizers.trig_canon import tan_canon

MIN_INIT_LOG = 1e-3
MIN_INIT_POWER = 1e-4
HALF_PI = np.pi / 2


def _make_wide_bounds_arg():
    """Create an expression whose get_bounds() returns (-inf, inf).

    Subtracting two nonneg variables gives bounds (-inf, inf), which
    exercises the canonicalizer's domain-clamping logic (e.g. clamping
    lb to 0 for log/entr) without artificially collapsing the feasible set.
    """
    a = Variable(nonneg=True)
    b = Variable(nonneg=True)
    return a - b


def _get_slack_vars(constraints):
    """Extract the slack variables (LHS of equality constraints) from canon output."""
    return [c.args[0] for c in constraints]


# ---------------------------------------------------------------------------
# Single-arg canonicalizers: log, entr, power(0.5), tan
# ---------------------------------------------------------------------------

def _canon_log(x):
    return log_canon(cp.log(x), [x])


def _canon_entr(x):
    return entr_canon(cp.entr(x), [x])


def _canon_power_half(x):
    return power_canon(cp.power(x, 0.5), [x])


def _canon_tan(x):
    return tan_canon(cp.tan(x), [x])


SINGLE_ARG_CANONS = [
    pytest.param(_canon_log, id="log"),
    pytest.param(_canon_entr, id="entr"),
    pytest.param(_canon_power_half, id="power_half"),
    pytest.param(_canon_tan, id="tan"),
]

# Canonicalizers whose domain requires lb >= 0.
NONNEG_DOMAIN_CANONS = [
    pytest.param(_canon_log, id="log"),
    pytest.param(_canon_entr, id="entr"),
    pytest.param(_canon_power_half, id="power_half"),
]


@pytest.mark.parametrize("canon_fn", SINGLE_ARG_CANONS)
class TestSingleArgWideBounds:
    """Wide (-inf, inf) bounds must be clamped to valid slack bounds."""

    def test_no_nan_in_bounds(self, canon_fn):
        arg = _make_wide_bounds_arg()
        _, constraints = canon_fn(arg)
        for t in _get_slack_vars(constraints):
            lb, ub = t.get_bounds()
            assert not np.any(np.isnan(lb))
            assert not np.any(np.isnan(ub))

    def test_upper_not_collapsed(self, canon_fn):
        arg = _make_wide_bounds_arg()
        _, constraints = canon_fn(arg)
        for t in _get_slack_vars(constraints):
            _, ub = t.get_bounds()
            assert np.all(ub > MIN_INIT_POWER)


@pytest.mark.parametrize("canon_fn", NONNEG_DOMAIN_CANONS)
class TestNonnegDomainUnbounded:
    """Unbounded variables get lb >= 0 for nonneg-domain canonicalizers."""

    def test_lb_nonneg(self, canon_fn):
        x = Variable(2)
        _, constraints = canon_fn(x)
        for t in _get_slack_vars(constraints):
            lb, ub = t.get_bounds()
            assert np.all(lb >= 0.0)
            assert np.all(ub == np.inf)


# ---------------------------------------------------------------------------
# log-specific tests
# ---------------------------------------------------------------------------

class TestLogCanon:

    def test_bounded_variable(self):
        x = Variable(2, bounds=[2.0, 10.0])
        _, constraints = _canon_log(x)
        t = _get_slack_vars(constraints)[0]
        lb, ub = t.get_bounds()
        np.testing.assert_allclose(lb, 2.0)
        np.testing.assert_allclose(ub, 10.0)

    def test_value_initialization(self):
        x = Variable(2)
        x.value = np.array([0.0001, 5.0])
        _, constraints = _canon_log(x)
        t = _get_slack_vars(constraints)[0]
        assert t.value is not None
        assert t.value[0] >= MIN_INIT_LOG
        np.testing.assert_allclose(t.value[1], 5.0)

    def test_vector_per_element_bounds(self):
        x = Variable(3, bounds=[np.array([0.5, 1.0, 2.0]),
                                np.array([5.0, 10.0, 20.0])])
        _, constraints = _canon_log(x)
        t = _get_slack_vars(constraints)[0]
        lb, ub = t.get_bounds()
        np.testing.assert_allclose(lb, [0.5, 1.0, 2.0])
        np.testing.assert_allclose(ub, [5.0, 10.0, 20.0])


# ---------------------------------------------------------------------------
# Two-arg canonicalizers: rel_entr, quad_over_lin, kl_div
# ---------------------------------------------------------------------------

def _canon_rel_entr(x, y):
    return rel_entr_canon(cp.rel_entr(x, y), [x, y])


def _canon_quad_over_lin(x, y):
    return quad_over_lin_canon(cp.quad_over_lin(x, y), [x, y])


def _canon_kl_div(x, y):
    expr = cp.kl_div(x, y)
    return kl_div_canon(expr, expr.args)


TWO_ARG_CANONS = [
    pytest.param(_canon_rel_entr, id="rel_entr"),
    pytest.param(_canon_quad_over_lin, id="quad_over_lin"),
    pytest.param(_canon_kl_div, id="kl_div"),
]


@pytest.mark.parametrize("canon_fn", TWO_ARG_CANONS)
class TestTwoArgWideBounds:
    """Wide (-inf, inf) bounds must be clamped to valid slack bounds."""

    def test_no_nan_in_bounds(self, canon_fn):
        arg = _make_wide_bounds_arg()
        y = Variable()
        _, constraints = canon_fn(arg, y)
        for t in _get_slack_vars(constraints):
            lb, ub = t.get_bounds()
            assert not np.any(np.isnan(lb))
            assert not np.any(np.isnan(ub))

    def test_upper_not_collapsed(self, canon_fn):
        arg = _make_wide_bounds_arg()
        y = Variable()
        _, constraints = canon_fn(arg, y)
        for t in _get_slack_vars(constraints):
            _, ub = t.get_bounds()
            assert np.all(ub > MIN_INIT_POWER)


class TestRelEntrCanon:

    def test_both_unbounded(self):
        x = Variable(2)
        y = Variable(2)
        _, constraints = _canon_rel_entr(x, y)
        for t in _get_slack_vars(constraints):
            lb, ub = t.get_bounds()
            assert np.all(lb >= 0.0)
            assert np.all(ub == np.inf)


class TestQuadOverLinCanon:

    def test_unbounded_denominator(self):
        x = Variable(2)
        y = Variable()
        _, constraints = _canon_quad_over_lin(x, y)
        t2 = _get_slack_vars(constraints)[-1]
        lb, ub = t2.get_bounds()
        assert np.all(lb >= 0.0)
        assert np.all(ub >= lb)


# ---------------------------------------------------------------------------
# tan-specific tests
# ---------------------------------------------------------------------------

class TestTanCanon:

    def test_unbounded_clips_to_half_pi(self):
        x = Variable(2)
        _, constraints = _canon_tan(x)
        t = _get_slack_vars(constraints)[0]
        lb, ub = t.get_bounds()
        np.testing.assert_allclose(lb, -HALF_PI)
        np.testing.assert_allclose(ub, HALF_PI)

    def test_tighter_bounds_preserved(self):
        x = Variable(2, bounds=[-0.5, 0.5])
        _, constraints = _canon_tan(x)
        t = _get_slack_vars(constraints)[0]
        lb, ub = t.get_bounds()
        np.testing.assert_allclose(lb, -0.5)
        np.testing.assert_allclose(ub, 0.5)

    def test_wide_bounds_defaults_to_half_pi(self):
        arg = _make_wide_bounds_arg()
        _, constraints = _canon_tan(arg)
        t = _get_slack_vars(constraints)[0]
        lb, ub = t.get_bounds()
        np.testing.assert_allclose(float(lb), -HALF_PI)
        np.testing.assert_allclose(float(ub), HALF_PI)


# ---------------------------------------------------------------------------
# power-specific tests
# ---------------------------------------------------------------------------

class TestPowerCanon:

    def test_integer_power_no_bounds_needed(self):
        x = Variable(2)
        _, constraints = power_canon(cp.power(x, 2), [x])
        assert len(constraints) == 0
