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


def _make_nan_bounds_arg():
    """Create an expression whose get_bounds() returns (NaN, NaN).

    Multiplying a nonneg variable [0, inf] by a nonpos variable [-inf, 0]
    produces 0*inf = NaN in the bound computation.
    """
    a = Variable(nonneg=True)
    b = Variable(nonpos=True)
    return a * b


class TestLogCanonBounds:

    def test_unbounded_variable(self):
        x = Variable(2)
        expr = cp.log(x)
        _, constraints = log_canon(expr, [x])
        t = constraints[0].args[0]
        lb, ub = t.get_bounds()
        assert np.all(lb >= 0.0)
        assert np.all(ub == np.inf)

    def test_bounded_variable(self):
        x = Variable(2, bounds=[2.0, 10.0])
        expr = cp.log(x)
        _, constraints = log_canon(expr, [x])
        t = constraints[0].args[0]
        lb, ub = t.get_bounds()
        np.testing.assert_allclose(lb, 2.0)
        np.testing.assert_allclose(ub, 10.0)

    def test_nan_bounds_no_nan_in_result(self):
        """NaN from get_bounds must not leak into slack variable bounds."""
        arg = _make_nan_bounds_arg()
        expr = cp.log(arg)
        _, constraints = log_canon(expr, expr.args)
        t = constraints[0].args[0]
        lb, ub = t.get_bounds()
        assert not np.any(np.isnan(lb)), f"NaN in lower bound: {lb}"
        assert not np.any(np.isnan(ub)), f"NaN in upper bound: {ub}"

    def test_nan_bounds_upper_not_collapsed(self):
        """Upper bound must not collapse to MIN_INIT when arg bounds are NaN."""
        arg = _make_nan_bounds_arg()
        expr = cp.log(arg)
        _, constraints = log_canon(expr, expr.args)
        t = constraints[0].args[0]
        lb, ub = t.get_bounds()
        assert np.all(ub > MIN_INIT_LOG), (
            f"Upper bound {ub} collapsed to MIN_INIT — should be +inf"
        )

    def test_value_initialization(self):
        x = Variable(2)
        x.value = np.array([0.0001, 5.0])
        expr = cp.log(x)
        _, constraints = log_canon(expr, [x])
        t = constraints[0].args[0]
        assert t.value is not None
        assert t.value[0] >= MIN_INIT_LOG
        np.testing.assert_allclose(t.value[1], 5.0)

    def test_vector_per_element_bounds(self):
        x = Variable(3, bounds=[np.array([0.5, 1.0, 2.0]),
                                np.array([5.0, 10.0, 20.0])])
        expr = cp.log(x)
        _, constraints = log_canon(expr, [x])
        t = constraints[0].args[0]
        lb, ub = t.get_bounds()
        np.testing.assert_allclose(lb, [0.5, 1.0, 2.0])
        np.testing.assert_allclose(ub, [5.0, 10.0, 20.0])


class TestEntrCanonBounds:

    def test_unbounded_variable(self):
        x = Variable(2)
        expr = cp.entr(x)
        _, constraints = entr_canon(expr, [x])
        t = constraints[0].args[0]
        lb, ub = t.get_bounds()
        assert np.all(lb >= 0.0)
        assert np.all(ub == np.inf)

    def test_nan_bounds_no_nan_in_result(self):
        arg = _make_nan_bounds_arg()
        expr = cp.entr(arg)
        _, constraints = entr_canon(expr, expr.args)
        t = constraints[0].args[0]
        lb, ub = t.get_bounds()
        assert not np.any(np.isnan(lb)), f"NaN in lower bound: {lb}"
        assert not np.any(np.isnan(ub)), f"NaN in upper bound: {ub}"

    def test_nan_bounds_upper_not_collapsed(self):
        arg = _make_nan_bounds_arg()
        expr = cp.entr(arg)
        _, constraints = entr_canon(expr, expr.args)
        t = constraints[0].args[0]
        lb, ub = t.get_bounds()
        assert np.all(ub > MIN_INIT_LOG), (
            f"Upper bound {ub} collapsed to MIN_INIT — should be +inf"
        )


class TestRelEntrCanonBounds:

    def test_both_unbounded(self):
        x = Variable(2)
        y = Variable(2)
        expr = cp.rel_entr(x, y)
        _, constraints = rel_entr_canon(expr, [x, y])
        t1 = constraints[0].args[0]
        t2 = constraints[1].args[0]
        for t in [t1, t2]:
            lb, ub = t.get_bounds()
            assert np.all(lb >= 0.0)
            assert np.all(ub == np.inf)

    def test_nan_bounds_no_nan_in_result(self):
        arg = _make_nan_bounds_arg()
        y = Variable()
        expr = cp.rel_entr(arg, y)
        _, constraints = rel_entr_canon(expr, expr.args)
        for c in constraints:
            t = c.args[0]
            lb, ub = t.get_bounds()
            assert not np.any(np.isnan(lb)), f"NaN in lower bound: {lb}"
            assert not np.any(np.isnan(ub)), f"NaN in upper bound: {ub}"

    def test_nan_bounds_upper_not_collapsed(self):
        arg = _make_nan_bounds_arg()
        y = Variable()
        expr = cp.rel_entr(arg, y)
        _, constraints = rel_entr_canon(expr, expr.args)
        for c in constraints:
            t = c.args[0]
            lb, ub = t.get_bounds()
            assert np.all(ub > MIN_INIT_LOG), (
                f"Upper bound {ub} collapsed to MIN_INIT — should be +inf"
            )


class TestQuadOverLinCanonBounds:

    def test_unbounded_denominator(self):
        x = Variable(2)
        y = Variable()
        expr = cp.quad_over_lin(x, y)
        _, constraints = quad_over_lin_canon(expr, [x, y])
        t2 = constraints[-1].args[0]
        lb, ub = t2.get_bounds()
        assert np.all(lb >= 0.0)
        assert np.all(ub >= lb)

    def test_nan_bounds_no_nan_in_result(self):
        x = Variable(2)
        denom = _make_nan_bounds_arg()
        expr = cp.quad_over_lin(x, denom)
        _, constraints = quad_over_lin_canon(expr, [x, denom])
        t2 = constraints[-1].args[0]
        lb, ub = t2.get_bounds()
        assert not np.any(np.isnan(lb)), f"NaN in lower bound: {lb}"
        assert not np.any(np.isnan(ub)), f"NaN in upper bound: {ub}"

    def test_nan_bounds_upper_not_collapsed(self):
        x = Variable(2)
        denom = _make_nan_bounds_arg()
        expr = cp.quad_over_lin(x, denom)
        _, constraints = quad_over_lin_canon(expr, [x, denom])
        t2 = constraints[-1].args[0]
        lb, ub = t2.get_bounds()
        assert np.all(ub > MIN_INIT_POWER), (
            f"Upper bound {ub} collapsed to MIN_INIT — should be +inf"
        )


class TestPowerCanonBounds:

    def test_fractional_power_unbounded(self):
        x = Variable(2)
        expr = cp.power(x, 0.5)
        _, constraints = power_canon(expr, [x])
        t = constraints[0].args[0]
        lb, ub = t.get_bounds()
        assert np.all(lb >= 0.0)
        assert np.all(ub == np.inf)

    def test_fractional_power_nan_bounds_no_nan(self):
        arg = _make_nan_bounds_arg()
        expr = cp.power(arg, 0.5)
        _, constraints = power_canon(expr, expr.args)
        t = constraints[0].args[0]
        lb, ub = t.get_bounds()
        assert not np.any(np.isnan(lb)), f"NaN in lower bound: {lb}"
        assert not np.any(np.isnan(ub)), f"NaN in upper bound: {ub}"

    def test_fractional_power_nan_bounds_upper_not_collapsed(self):
        arg = _make_nan_bounds_arg()
        expr = cp.power(arg, 0.5)
        _, constraints = power_canon(expr, expr.args)
        t = constraints[0].args[0]
        lb, ub = t.get_bounds()
        assert np.all(ub > MIN_INIT_POWER), (
            f"Upper bound {ub} collapsed to MIN_INIT — should be +inf"
        )

    def test_integer_power_no_bounds_needed(self):
        x = Variable(2)
        expr = cp.power(x, 2)
        _, constraints = power_canon(expr, [x])
        assert len(constraints) == 0


class TestTanCanonBounds:

    def test_unbounded_clips_to_half_pi(self):
        x = Variable(2)
        expr = cp.tan(x)
        _, constraints = tan_canon(expr, [x])
        t = constraints[0].args[0]
        lb, ub = t.get_bounds()
        np.testing.assert_allclose(lb, -HALF_PI)
        np.testing.assert_allclose(ub, HALF_PI)

    def test_tighter_bounds_preserved(self):
        x = Variable(2, bounds=[-0.5, 0.5])
        expr = cp.tan(x)
        _, constraints = tan_canon(expr, [x])
        t = constraints[0].args[0]
        lb, ub = t.get_bounds()
        np.testing.assert_allclose(lb, -0.5)
        np.testing.assert_allclose(ub, 0.5)

    def test_nan_bounds_no_nan_in_result(self):
        arg = _make_nan_bounds_arg()
        expr = cp.tan(arg)
        _, constraints = tan_canon(expr, expr.args)
        t = constraints[0].args[0]
        lb, ub = t.get_bounds()
        assert not np.any(np.isnan(lb)), f"NaN in lower bound: {lb}"
        assert not np.any(np.isnan(ub)), f"NaN in upper bound: {ub}"

    def test_nan_bounds_defaults_to_half_pi(self):
        arg = _make_nan_bounds_arg()
        expr = cp.tan(arg)
        _, constraints = tan_canon(expr, expr.args)
        t = constraints[0].args[0]
        lb, ub = t.get_bounds()
        np.testing.assert_allclose(float(lb), -HALF_PI)
        np.testing.assert_allclose(float(ub), HALF_PI)


class TestKLDivCanonBounds:

    def test_nan_bounds_no_nan_in_result(self):
        arg = _make_nan_bounds_arg()
        y = Variable()
        expr = cp.kl_div(arg, y)
        _, constraints = kl_div_canon(expr, expr.args)
        for c in constraints:
            for var in c.variables():
                lb, ub = var.get_bounds()
                assert not np.any(np.isnan(lb)), f"NaN in lower bound of {var}"
                assert not np.any(np.isnan(ub)), f"NaN in upper bound of {var}"
