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
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests.nlp_tests.derivative_checker import DerivativeChecker


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestScalarProblems():

    def test_exp(self):
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.exp(x)), [x >= 4])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_entropy(self):
        x = cp.Variable()
        prob = cp.Problem(cp.Maximize(cp.entr(x)), [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_KL(self):
       p = cp.Variable()
       q = cp.Variable()
       prob = cp.Problem(cp.Minimize(cp.kl_div(p, q)), [p >= 0.1, q >= 0.1, p + q == 2])
       prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
       assert prob.status == cp.OPTIMAL
       checker = DerivativeChecker(prob)
       checker.run_and_assert()

    def test_KL_matrix(self):
        Y = cp.Variable((3, 3))
        X = cp.Variable((3, 3))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.kl_div(X, Y))),
                        [X >= 0.1, Y >= 0.1, cp.sum(X) + cp.sum(Y) == 6])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_entropy_matrix(self):
        x = cp.Variable((3, 2))
        prob = cp.Problem(cp.Maximize(cp.sum(cp.entr(x))), [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_logistic(self):
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.logistic(x)), [x >= 0.4])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_logistic_matrix(self):
        x = cp.Variable((3, 2))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.logistic(x))), [x >= 0.4])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_power(self):
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.power(x, 3)), [x >= 2])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_power_matrix(self):
        x = cp.Variable((3, 2))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.power(x, 3))), [x >= 2])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_power_fractional(self):
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.power(x, 1.5)), [x >= 2])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

        x = cp.Variable((3, 2))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.power(x, 0.6))), [x >= 2])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_power_fractional_matrix(self):
        x = cp.Variable((3, 2))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.power(x, 1.5))), [x >= 2])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

        x = cp.Variable((3, 2))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.power(x, 0.6))), [x >= 2])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_scalar_trig(self):
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.nlp.tan(x)), [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

        prob = cp.Problem(cp.Minimize(cp.nlp.sin(x)), [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

        prob = cp.Problem(cp.Minimize(cp.nlp.cos(x)), [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

        prob = cp.Problem(cp.Minimize(cp.nlp.atan(x)), [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_matrix_trig(self):
        x = cp.Variable((3, 2))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.nlp.tan(x))), [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

        prob = cp.Problem(cp.Minimize(cp.sum(cp.nlp.sin(x))), [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

        prob = cp.Problem(cp.Minimize(cp.sum(cp.nlp.cos(x))), [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

        prob = cp.Problem(cp.Minimize(cp.sum(cp.nlp.atan(x))), [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_matrix_hyperbolic(self):
        x = cp.Variable((3, 2))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.nlp.sinh(x))), [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

        prob = cp.Problem(cp.Minimize(cp.sum(cp.nlp.tanh(x))), [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_atan_composite(self):
        # atan of a non-variable argument exercises the engine's chain rule
        n = 10
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.nlp.atan(cp.logistic(x * 3)))),
                          [x >= 0.1, cp.sum(x) == 10])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_scalar_hyperbolic(self):
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.nlp.sinh(x)), [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

        prob = cp.Problem(cp.Minimize(cp.nlp.tanh(x)), [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_xexp(self):
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.xexp(x)), [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

        x = cp.Variable((3, 2))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.xexp(x))), [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_scalar_quad_form(self):
        x = cp.Variable((1, ))
        P = np.array([[3]])
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P)), [x >= 1])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_scalar_quad_over_lin(self):
        x = cp.Variable()
        y = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.quad_over_lin(x, y)), [x >= 1, y <= 1])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_matrix_quad_over_lin(self):
        x = cp.Variable((3, 2))
        y = cp.Variable((1, ))
        prob = cp.Problem(cp.Minimize(cp.quad_over_lin(x, y)), [x >= 1, y <= 1])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

        y = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.quad_over_lin(x, y)), [x >= 1, y <= 1])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_rel_entr_both_scalar_variables(self):
        x = cp.Variable()
        y = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.rel_entr(x, y)),
                          [x >= 0.1, y >= 0.1, x <= 2, y <= 2])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

        x = cp.Variable((1, ))
        y = cp.Variable((1, ))
        prob = cp.Problem(cp.Minimize(cp.rel_entr(x, y)),
                          [x >= 0.1, y >= 0.1, x <= 2, y <= 2])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_rel_entr_matrix_variable_and_scalar_variable(self):
        x = cp.Variable((3, 2))
        y = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.sum(cp.rel_entr(x, y))),
                          [x >= 0.1, y >= 0.1, x <= 2, y <= 2])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_rel_entr_scalar_variable_and_matrix_variable(self):
        x = cp.Variable()
        y = cp.Variable((3, 2))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.rel_entr(x, y))),
                          [x >= 0.1, y >= 0.1, x <= 2, y <= 2])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_rel_entr_both_matrix_variables(self):
        x = cp.Variable((3, 2))
        y = cp.Variable((3, 2))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.rel_entr(x, y))),
                          [x >= 0.1, y >= 0.1, x <= 2, y <= 2])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_rel_entr_both_vector_variables(self):
        x = cp.Variable((3, ))
        y = cp.Variable((3, ))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.rel_entr(x, y))),
                          [x >= 0.1, y >= 0.1, x <= 2, y <= 2])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    # The atan2 tests leave the variables without initial values, so they
    # start at the origin and exercise the canonicalizer's nudge away from it.
    def test_atan2_both_scalar_variables(self):
        x = cp.Variable()
        y = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.nlp.atan2(x, y)),
                          [x >= 0.1, y >= 0.1, x <= 2, y <= 2])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        assert np.isclose(prob.value, np.arctan2(0.1, 2))
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

        x = cp.Variable((1, ))
        y = cp.Variable((1, ))
        prob = cp.Problem(cp.Minimize(cp.nlp.atan2(x, y)),
                          [x >= 0.1, y >= 0.1, x <= 2, y <= 2])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_atan2_matrix_variable_and_scalar_variable(self):
        x = cp.Variable((3, 2))
        y = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.sum(cp.nlp.atan2(x, y))),
                          [x >= 0.1, y >= 0.1, x <= 2, y <= 2])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_atan2_scalar_variable_and_matrix_variable(self):
        x = cp.Variable()
        y = cp.Variable((3, 2))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.nlp.atan2(x, y))),
                          [x >= 0.1, y >= 0.1, x <= 2, y <= 2])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_atan2_both_matrix_variables(self):
        x = cp.Variable((3, 2))
        y = cp.Variable((3, 2))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.nlp.atan2(x, y))),
                          [x >= 0.1, y >= 0.1, x <= 2, y <= 2])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_atan2_both_vector_variables(self):
        x = cp.Variable((3, ))
        y = cp.Variable((3, ))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.nlp.atan2(x, y))),
                          [x >= 0.1, y >= 0.1, x <= 2, y <= 2])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_atan2_same_variable(self):
        # atan2(x, x) = pi/4 for x > 0; the canonicalizer lifts both slots to
        # distinct variables so the leaf-only engine atom accepts them.
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - 1) + cp.sum(cp.nlp.atan2(x, x))),
                          [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        assert np.isclose(prob.value, 3 * np.pi / 4)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_atan2_angle_wrap(self):
        # atan2(sin(theta), cos(theta)) wraps theta into (-pi, pi]. The
        # arguments are not variables, so the canonicalizer has to lift them.
        target = np.array([0.5, -2.0, 2.5])
        theta = cp.Variable(3)
        theta.value = target + 2 * np.pi
        wrapped = cp.nlp.atan2(cp.nlp.sin(theta), cp.nlp.cos(theta))
        prob = cp.Problem(cp.Minimize(cp.sum_squares(wrapped - target)))
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
        assert prob.status == cp.OPTIMAL
        np.testing.assert_allclose(
            np.arctan2(np.sin(theta.value), np.cos(theta.value)), target, atol=1e-5)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
