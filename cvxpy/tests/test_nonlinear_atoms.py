"""
Copyright 2013 Steven Diamond, Eric Chu

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

import math

import numpy as np
import pytest

import cvxpy as cp
from cvxpy.tests.base_test import BaseTest


class TestNonlinearAtoms(BaseTest):
    """ Unit tests for the nonlinear atoms module. """

    def setUp(self) -> None:
        self.x = cp.Variable(2, name='x')
        self.y = cp.Variable(2, name='y')

        self.A = cp.Variable((2, 2), name='A')
        self.B = cp.Variable((2, 2), name='B')
        self.C = cp.Variable((3, 2), name='C')

    def test_log_problem(self) -> None:
        # Log in objective.
        obj = cp.Maximize(cp.sum(cp.log(self.x)))
        constr = [self.x <= [1, math.e]]
        p = cp.Problem(obj, constr)
        result = p.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(result, 1)
        self.assertItemsAlmostEqual(self.x.value, [1, math.e])

        # Log in constraint.
        obj = cp.Minimize(cp.sum(self.x))
        constr = [cp.log(self.x) >= 0, self.x <= [1, 1]]
        p = cp.Problem(obj, constr)
        result = p.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(result, 2)
        self.assertItemsAlmostEqual(self.x.value, [1, 1])

        # Index into log.
        obj = cp.Maximize(cp.log(self.x)[1])
        constr = [self.x <= [1, math.e]]
        p = cp.Problem(obj, constr)
        result = p.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(result, 1)

        # Scalar log.
        obj = cp.Maximize(cp.log(self.x[1]))
        constr = [self.x <= [1, math.e]]
        p = cp.Problem(obj, constr)
        result = p.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(result, 1)

    def test_entr(self) -> None:
        """Test the entr atom.
        """
        self.assertEqual(cp.entr(0).value, 0)
        assert np.isneginf(cp.entr(-1).value)

    def test_kl_div(self) -> None:
        """Test a problem with kl_div.
        """
        kK = 50
        kSeed = 10

        prng = np.random.RandomState(kSeed)
        # Generate a random reference distribution
        npSPriors = prng.uniform(0.0, 1.0, kK)
        npSPriors = npSPriors / sum(npSPriors)

        # Reference distribution
        p_refProb = cp.Parameter(kK, nonneg=True)
        # Distribution to be estimated
        v_prob = cp.Variable(kK)
        objkl = cp.sum(cp.kl_div(v_prob, p_refProb))

        constrs = [cp.sum(v_prob) == 1]
        klprob = cp.Problem(cp.Minimize(objkl), constrs)
        p_refProb.value = npSPriors
        klprob.solve(solver=cp.SCS)
        self.assertItemsAlmostEqual(v_prob.value, npSPriors, places=3)
        klprob.solve(solver=cp.CLARABEL)
        self.assertItemsAlmostEqual(v_prob.value, npSPriors, places=3)

    def test_rel_entr(self) -> None:
        """Test a problem with rel_entr.
        """
        kK = 50
        kSeed = 10

        prng = np.random.RandomState(kSeed)
        # Generate a random reference distribution
        npSPriors = prng.uniform(0.0, 1.0, kK)
        npSPriors = npSPriors / sum(npSPriors)

        # Reference distribution
        p_refProb = cp.Parameter(kK, nonneg=True)
        # Distribution to be estimated
        v_prob = cp.Variable(kK)
        obj_rel_entr = cp.sum(cp.rel_entr(v_prob, p_refProb))

        constrs = [cp.sum(v_prob) == 1]
        rel_entr_prob = cp.Problem(cp.Minimize(obj_rel_entr), constrs)
        p_refProb.value = npSPriors
        rel_entr_prob.solve(solver=cp.SCS)
        self.assertItemsAlmostEqual(v_prob.value, npSPriors, places=3)
        rel_entr_prob.solve(solver=cp.CLARABEL)
        self.assertItemsAlmostEqual(v_prob.value, npSPriors, places=3)

    def test_difference_kl_div_rel_entr(self) -> None:
        """A test showing the difference between kl_div and rel_entr
        """
        x = cp.Variable()
        y = cp.Variable()

        kl_div_prob = cp.Problem(cp.Minimize(cp.kl_div(x, y)), constraints=[x + y <= 1])
        kl_div_prob.solve(solver=cp.CLARABEL)
        self.assertItemsAlmostEqual(x.value, y.value, places=3)
        self.assertItemsAlmostEqual(kl_div_prob.value, 0)

        rel_entr_prob = cp.Problem(cp.Minimize(cp.rel_entr(x, y)), constraints=[x + y <= 1])
        rel_entr_prob.solve(solver=cp.CLARABEL)

        """
        Reference solution computed by passing the following command to Wolfram Alpha:
        minimize x*log(x/y) subject to {x + y <= 1, 0 <= x, 0 <= y}
        """
        self.assertItemsAlmostEqual(x.value, 0.2178117, places=4)
        self.assertItemsAlmostEqual(y.value, 0.7821882, places=4)
        self.assertItemsAlmostEqual(rel_entr_prob.value, -0.278464)

    def test_entr_prob(self) -> None:
        """Test a problem with entr.
        """
        for n in [5, 10, 25]:
            x = cp.Variable(n)
            obj = cp.Maximize(cp.sum(cp.entr(x)))
            p = cp.Problem(obj, [cp.sum(x) == 1])
            p.solve(solver=cp.CLARABEL)
            self.assertItemsAlmostEqual(x.value, n*[1./n], places=3)
            p.solve(solver=cp.SCS)
            self.assertItemsAlmostEqual(x.value, n*[1./n], places=3)

    def test_exp(self) -> None:
        """Test a problem with exp.
        """
        for n in [5, 10, 25]:
            x = cp.Variable(n)
            obj = cp.Minimize(cp.sum(cp.exp(x)))
            p = cp.Problem(obj, [cp.sum(x) == 1])
            p.solve(solver=cp.SCS)
            self.assertItemsAlmostEqual(x.value, n*[1./n], places=3)
            p.solve(solver=cp.CLARABEL)
            self.assertItemsAlmostEqual(x.value, n*[1./n], places=4)

    def test_log(self) -> None:
        """Test a problem with log.
        """
        for n in [5, 10, 25]:
            x = cp.Variable(n)
            obj = cp.Maximize(cp.sum(cp.log(x)))
            p = cp.Problem(obj, [cp.sum(x) == 1])
            p.solve(solver=cp.CLARABEL)
            self.assertItemsAlmostEqual(x.value, n*[1./n])
            p.solve(solver=cp.SCS)
            self.assertItemsAlmostEqual(x.value, n*[1./n], places=2)


@pytest.mark.parametrize(
    ("atom", "numeric", "grad", "domain_size"),
    [
        (cp.nlp.sin, np.sin, np.cos, 0),
        (cp.nlp.cos, np.cos, lambda x: -np.sin(x), 0),
        (cp.nlp.tan, np.tan, lambda x: 1 / np.cos(x) ** 2, 2),
    ],
)
def test_trig_atoms_metadata_numeric_and_grad(atom, numeric, grad, domain_size):
    x = cp.Variable(2)
    expr = atom(x)
    value = np.array([0.2, -0.3])

    np.testing.assert_allclose(expr.numeric([value]), numeric(value))
    assert expr.sign_from_args() == (False, False)
    assert not expr.is_atom_convex()
    assert not expr.is_atom_concave()
    assert expr.is_atom_smooth()
    assert not expr.is_incr(0)
    assert not expr.is_decr(0)
    assert len(expr._domain()) == domain_size

    grad_matrix = expr._grad([value])[0]
    np.testing.assert_allclose(grad_matrix.diagonal(), grad(value))


def test_atan_metadata_numeric_and_grad():
    x = cp.Variable(2)
    expr = cp.nlp.atan(x)
    value = np.array([0.2, -0.3])

    np.testing.assert_allclose(expr.numeric([value]), np.arctan(value))
    assert expr.sign_from_args() == (False, False)
    assert cp.nlp.atan(cp.Variable(2, nonneg=True)).sign_from_args() == (True, False)
    assert cp.nlp.atan(cp.Variable(2, nonpos=True)).sign_from_args() == (False, True)
    assert not expr.is_atom_convex()
    assert not expr.is_atom_concave()
    assert expr.is_atom_smooth()
    assert expr.is_incr(0)
    assert not expr.is_decr(0)
    assert len(expr._domain()) == 0

    grad_matrix = expr._grad([value])[0]
    np.testing.assert_allclose(grad_matrix.diagonal(), 1 / (1 + value**2))


def test_atan2_metadata_numeric_and_grad():
    y = cp.Variable(2)
    x = cp.Variable(2)
    expr = cp.nlp.atan2(y, x)
    y_val = np.array([0.2, -0.3])
    x_val = np.array([-0.5, 0.4])

    np.testing.assert_allclose(expr.numeric([y_val, x_val]), np.arctan2(y_val, x_val))
    assert cp.nlp.atan2(cp.Variable(), cp.Variable((3, 2))).shape == (3, 2)
    assert expr.sign_from_args() == (False, False)
    assert cp.nlp.atan2(cp.Variable(2, nonneg=True), x).sign_from_args() == (True, False)
    assert cp.nlp.atan2(cp.Variable(2, nonpos=True), x).sign_from_args() == (False, False)
    assert cp.nlp.atan2(cp.Variable(2, nonpos=True),
                        cp.Variable(2, nonneg=True)).sign_from_args() == (False, True)
    assert not expr.is_atom_convex()
    assert not expr.is_atom_concave()
    assert expr.is_atom_smooth()
    assert not expr.is_incr(0) and not expr.is_incr(1)
    assert not expr.is_decr(0) and not expr.is_decr(1)
    assert len(expr._domain()) == 0

    r2 = y_val**2 + x_val**2
    grad_y, grad_x = expr._grad([y_val, x_val])
    np.testing.assert_allclose(grad_y.diagonal(), x_val / r2)
    np.testing.assert_allclose(grad_x.diagonal(), -y_val / r2)
    assert expr._grad([np.array([0.0, 1.0]), np.array([0.0, 1.0])]) == [None, None]


def test_atan2_canon_lifts_to_fresh_variables():
    from cvxpy.reductions.dnlp2smooth.canonicalizers.atan2_canon import MIN_INIT_RADIUS
    from cvxpy.reductions.dnlp2smooth.dnlp2smooth import Dnlp2Smooth

    # Same variable in both slots, starting at the origin.
    x = cp.Variable(3)
    x.value = np.zeros(3)
    prob = cp.Problem(cp.Minimize(cp.sum(cp.nlp.atan2(x, x))))
    canon, _ = Dnlp2Smooth().apply(prob)
    atom = canon.objective.expr.args[0]
    t1, t2 = atom.args
    assert isinstance(t1, cp.Variable) and isinstance(t2, cp.Variable)
    assert t1 is not t2 and t1 is not x and t2 is not x
    assert t1.shape == (3,) and t2.shape == (3,)
    assert len(canon.constraints) == 2
    np.testing.assert_allclose(np.hypot(t2.value, t1.value), MIN_INIT_RADIUS)
    np.testing.assert_array_equal(x.value, np.zeros(3))

    # Scalar and matrix arguments are lifted to variables of the atom's shape,
    # and initial values away from the origin are kept.
    y = cp.Variable()
    y.value = 2.0
    z = cp.Variable((3, 2))
    z.value = np.full((3, 2), -1.5)
    prob = cp.Problem(cp.Minimize(cp.sum(cp.nlp.atan2(y, z))))
    canon, _ = Dnlp2Smooth().apply(prob)
    t1, t2 = canon.objective.expr.args[0].args
    assert t1.shape == (3, 2) and t2.shape == (3, 2)
    np.testing.assert_allclose(t1.value, np.full((3, 2), 2.0))
    np.testing.assert_allclose(t2.value, np.full((3, 2), -1.5))


@pytest.mark.parametrize(
    ("atom", "numeric", "domain_size"),
    [
        (cp.nlp.sinh, np.sinh, 0),
        (cp.nlp.tanh, np.tanh, 0),
        (cp.nlp.asinh, np.arcsinh, 0),
        (cp.nlp.atanh, np.arctanh, 2),
    ],
)
def test_hyperbolic_atoms_metadata_numeric_and_unimplemented_grad(atom, numeric, domain_size):
    x = cp.Variable(2)
    expr = atom(x)
    value = np.array([0.2, -0.3])

    np.testing.assert_allclose(expr.numeric([value]), numeric(value))
    assert expr.sign_from_args() == (False, False)
    assert not expr.is_atom_convex()
    assert not expr.is_atom_concave()
    assert expr.is_atom_smooth()
    assert expr.is_incr(0)
    assert not expr.is_decr(0)
    assert len(expr._domain()) == domain_size
    with pytest.raises(NotImplementedError):
        expr._grad([value])
