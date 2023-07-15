"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import pytest

import cvxpy as cp
from cvxpy.atoms.perspective import perspective
from cvxpy.constraints.exponential import ExpCone


def test_monotonicity():
    x = cp.Variable(nonneg=True)
    f = cp.exp(x)
    s = cp.Variable(nonneg=True)
    p = perspective(f, s)

    assert p.is_nonneg()
    assert not p.is_nonpos()

    assert not p.is_incr(0)
    assert not p.is_incr(1)

    assert not p.is_decr(0)
    assert not p.is_decr(1)


@pytest.fixture(params=[2, 3, 4, -2, 0])
def quad_example(request):
    # Reference expected output
    x = cp.Variable()
    s = cp.Variable()

    r = request.param

    obj = cp.quad_over_lin(x, s) + r*x - 4*s
    constraints = [x >= 2, s <= .5]
    prob_ref = cp.Problem(cp.Minimize(obj), constraints)
    prob_ref.solve(solver=cp.ECOS)

    return prob_ref.value, s.value, x.value, r


@pytest.mark.parametrize("p", [1, 2])
def test_p_norms(p):
    x = cp.Variable(3)
    s = cp.Variable(nonneg=True, name='s')
    f = cp.norm(x, p)
    obj = cp.perspective(f, s)
    constraints = [1 == s, x >= [1, 2, 3]]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.ECOS)

    # reference problem
    ref_x = cp.Variable(3, pos=True)
    ref_s = cp.Variable(pos=True)

    obj = cp.sum(cp.power(ref_x, p) / cp.power(ref_s, p-1))

    ref_constraints = [ref_x >= [1, 2, 3], ref_s == 1]
    ref_prob = cp.Problem(cp.Minimize(obj), ref_constraints)
    ref_prob.solve(gp=True)

    assert np.isclose(prob.value**p, ref_prob.value)
    assert np.allclose(x.value, ref_x.value)
    if p != 1:  # s not used when denominator is s^0
        assert np.isclose(s.value, ref_s.value)


@pytest.mark.parametrize("cvx", [True, False])
def test_rel_entr(cvx):
    x = cp.Variable()
    s = cp.Variable(nonneg=True)
    f = cp.log(x)*(-1 if cvx else 1)
    obj = cp.perspective(f, s)
    constraints = [1 <= s, s <= 2, 1 <= x, x <= 2]
    prob = cp.Problem(cp.Minimize(obj) if cvx else cp.Maximize(obj), constraints)
    prob.solve(solver=cp.ECOS)

    # reference problem
    ref_x = cp.Variable()
    ref_s = cp.Variable()
    obj = cp.rel_entr(ref_s, ref_x) * (1 if cvx else -1)

    ref_constraints = [1 <= ref_x, ref_x <= 2, 1 <= ref_s, ref_s <= 2]
    ref_prob = cp.Problem(cp.Minimize(obj) if cvx else cp.Maximize(obj), ref_constraints)
    ref_prob.solve(solver=cp.ECOS)

    assert np.isclose(prob.value, ref_prob.value)
    assert np.allclose(x.value, ref_x.value)
    assert np.isclose(s.value, ref_s.value)


def test_exp():
    x = cp.Variable()
    s = cp.Variable(nonneg=True)
    f = cp.exp(x)
    obj = cp.perspective(f, s)
    constraints = [s >= 1, 1 <= x]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.ECOS)

    # reference problem
    ref_x = cp.Variable()
    ref_s = cp.Variable()
    ref_z = cp.Variable()

    obj = ref_z
    ref_constraints = [
        ExpCone(ref_x, ref_s, ref_z),
        ref_x >= 1, ref_s >= 1]
    ref_prob = cp.Problem(cp.Minimize(obj), ref_constraints)
    ref_prob.solve(solver=cp.ECOS)

    assert np.isclose(prob.value, ref_prob.value)
    assert np.isclose(x.value, ref_x.value)
    assert np.isclose(s.value, ref_s.value)


@pytest.fixture
def lse_example():
    # reference problem
    ref_x = cp.Variable(3)
    ref_s = cp.Variable()
    ref_z = cp.Variable(3)
    ref_t = cp.Variable()

    ref_constraints = [
        ref_s >= cp.sum(ref_z),
        [1, 2, 3] <= ref_x, 1 <= ref_s, ref_s <= 2]
    ref_constraints += [ExpCone(ref_x[i]-ref_t, ref_s, ref_z[i]) for i in range(3)]
    ref_prob = cp.Problem(cp.Minimize(ref_t), ref_constraints)
    ref_prob.solve(solver=cp.ECOS)

    return ref_prob.value, ref_x.value, ref_s.value


def test_lse(lse_example):
    x = cp.Variable(3)
    s = cp.Variable(nonneg=True)
    f = cp.log_sum_exp(x)

    obj = cp.perspective(f, s)
    constraints = [1 <= s, s <= 2, [1, 2, 3] <= x]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.ECOS)

    ref_prob, ref_x, ref_s = lse_example

    assert np.isclose(prob.value, ref_prob)
    assert np.allclose(x.value, ref_x)
    assert np.isclose(s.value, ref_s)


def test_lse_atom(lse_example):
    x = cp.Variable(3)
    s = cp.Variable(nonneg=True)
    f_exp = cp.log_sum_exp(x)

    obj = cp.perspective(f_exp, s)
    constraints = [1 <= s, s <= 2, [1, 2, 3] <= x]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.ECOS)

    # reference problem
    ref_prob, ref_x, ref_s = lse_example

    assert np.isclose(prob.value, ref_prob)
    assert np.allclose(x.value, ref_x)
    assert np.isclose(s.value, ref_s)


@pytest.mark.parametrize("x_val,s_val", [(1, 2), (5, .25), (.5, 7)])
def test_evaluate_persp(x_val, s_val):
    x = cp.Variable()
    s = cp.Variable(nonneg=True)
    f_exp = cp.square(x)+3*x-5
    obj = cp.perspective(f_exp, s)

    val_array = np.array([s_val, x_val])

    x.value = np.array(x_val)
    s.value = np.array(s_val)  # currently assumes variables have values before querrying
    val = obj.numeric(val_array)

    # true val
    ref_val = x_val**2/s_val + 3*x_val - 5*s_val

    assert np.isclose(val, ref_val)


def test_quad_atom(quad_example):
    ref_val, ref_s, ref_x, r = quad_example

    # form objective, introduce original variable
    x = cp.Variable()
    s = cp.Variable(nonneg=True)

    f_exp = cp.square(x) + r*x - 4

    obj = cp.perspective(f_exp, s)

    constraints = [s <= .5, x >= 2]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(verbose=True)

    assert np.isclose(prob.value, ref_val)
    assert np.isclose(x.value, ref_x)  # assuming the solutions are unique...
    assert np.isclose(s.value, ref_s)


def test_quad_persp_persp(quad_example):
    ref_val, ref_s, ref_x, r = quad_example

    # form objective, introduce original variable
    x = cp.Variable()
    s = cp.Variable(nonneg=True)
    t = cp.Variable(nonneg=True)

    f_exp = cp.square(x) + r*x - 4
    obj_inner = cp.perspective(f_exp, s)

    obj = cp.perspective(obj_inner, t)
    # f(x) -> sf(x/s) -> t(s/t)f(xt/ts) -> sf(x/s)

    constraints = [.1 <= s, s <= .5, x >= 2, .1 <= t, t <= .5]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(verbose=True)

    assert np.isclose(prob.value, ref_val)
    assert np.isclose(x.value, ref_x)  # assuming the solutions are unique...
    assert np.isclose(s.value, ref_s)


def test_quad_quad():
    # reference problem
    ref_x = cp.Variable()
    ref_s = cp.Variable(nonneg=True)

    # f(x) = x^4 -> persp(f)(x,s) = x^4 / s^3 = (x^2/s) ^2 / s
    obj = cp.quad_over_lin(cp.quad_over_lin(ref_x, ref_s), ref_s)
    constraints = [ref_x >= 5, ref_s <= 3]
    ref_prob = cp.Problem(cp.Minimize(obj), constraints)

    ref_prob.solve(solver=cp.ECOS)

    # perspective problem
    x = cp.Variable()
    s = cp.Variable(nonneg=True)
    f = cp.power(x, 4)
    obj = cp.perspective(f, s)

    constraints = [x >= 5, s <= 3]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.ECOS)

    assert np.isclose(prob.value, ref_prob.value)
    assert np.isclose(x.value, ref_x.value)
    assert np.isclose(s.value, ref_s.value)


@pytest.mark.parametrize("n", [4, 5, 7, 11])
def test_power(n):
    # reference problem
    ref_x = cp.Variable(pos=True)
    ref_s = cp.Variable(pos=True)

    # f(x) = x^n -> persp(f)(x,s) = x^n / s^(n-1)
    obj = cp.power(ref_x, n)/cp.power(ref_s, n-1)
    constraints = [ref_x >= 1, ref_s <= .5]
    ref_prob = cp.Problem(cp.Minimize(obj), constraints)

    ref_prob.solve(gp=True)

    # perspective problem
    x = cp.Variable()
    s = cp.Variable(nonneg=True)
    f = cp.power(x, n)
    obj = cp.perspective(f, s)

    constraints = [x >= 1, s <= .5]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.ECOS)

    assert np.isclose(prob.value, ref_prob.value)
    assert np.isclose(x.value, ref_x.value)
    assert np.isclose(s.value, ref_s.value)


def test_psd_tr_persp():
    # reference problem
    ref_P = cp.Variable((2, 2), PSD=True)

    obj = cp.trace(ref_P)
    constraints = [ref_P == np.eye(2)]
    ref_prob = cp.Problem(cp.Minimize(obj), constraints)

    ref_prob.solve(solver=cp.SCS)

    # perspective problem
    P = cp.Variable((2, 2), PSD=True)
    s = cp.Variable(nonneg=True)

    f = cp.trace(P)

    obj = cp.perspective(f, s)
    constraints = [P == np.eye(2), s == 1]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.SCS)

    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, ref_prob.value)


@pytest.mark.parametrize("n", [2, 3, 11])
def test_psd_mf_persp(n):
    # reference problem
    ref_x = cp.Variable(n)
    ref_P = cp.Variable((n, n), PSD=True)

    # matrix_frac is homogenous of degree 1 so its perspective is itself.
    obj = cp.matrix_frac(ref_x, ref_P)
    constraints = [ref_x == 5, ref_P == np.eye(n)]
    ref_prob = cp.Problem(cp.Minimize(obj), constraints)
    ref_prob.solve(solver=cp.SCS)

    # perspective problem
    x = cp.Variable(n)
    P = cp.Variable((n, n), PSD=True)
    s = cp.Variable(nonneg=True)

    f = cp.matrix_frac(x, P)
    obj = cp.perspective(f, s)
    constraints = [x == 5, P == np.eye(n), s == 1, ]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.SCS)

    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, ref_prob.value, atol=1e-2)
    assert np.allclose(x.value, ref_x.value, atol=1e-2)


@pytest.mark.parametrize("n", [2, 3, 11])
def test_psd_tr_square(n):
    # reference problem
    ref_s = cp.Variable(nonneg=True)
    ref_P = cp.Variable((n, n), PSD=True)

    # Tr(X)^2 perspective is quad over lin of Tr(X)
    obj = cp.quad_over_lin(cp.trace(ref_P), ref_s)
    constraints = [ref_s <= 5, ref_P >> np.eye(n)]
    ref_prob = cp.Problem(cp.Minimize(obj), constraints)
    ref_prob.solve(solver=cp.SCS)

    # perspective problem
    P = cp.Variable((n, n), PSD=True)
    s = cp.Variable(nonneg=True)

    f = cp.perspective(cp.square(cp.trace(P)), s)
    obj = cp.perspective(f, s)
    constraints = [s <= 5, P >> np.eye(n)]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.SCS)

    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, ref_prob.value, atol=1e-3)
    assert np.allclose(P.value, ref_P.value, atol=1e-4)


def test_diag():
    X_ref = cp.Variable((2, 2), diag=True)
    obj = cp.trace(X_ref)
    constraints = [cp.diag(X_ref) >= [1, 2]]
    ref_prob = cp.Problem(cp.Minimize(obj), constraints)
    ref_prob.solve()

    X = cp.Variable((2, 2), diag=True)
    f = cp.trace(X)
    s = cp.Variable(nonneg=True)
    obj = cp.perspective(f, s)
    constraints = [cp.diag(X) >= [1, 2], s == 1]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()

    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, ref_prob.value, atol=1e-3)
    assert np.allclose(X.value.toarray(), X_ref.value.toarray(), atol=1e-4)


def test_scalar_x():
    x = cp.Variable()
    s = cp.Variable(nonneg=True)
    obj = perspective(x-1, s)

    prob = cp.Problem(cp.Minimize(obj), [x >= 3.14, s <= 1])
    prob.solve()
    assert np.isclose(prob.value, 3.14 - 1)


def test_assert_s_nonzero():
    # If s=0 arises, make sure we ask for a recession function
    x = cp.Variable()
    s = cp.Variable(nonneg=True)
    obj = perspective(x+1, s)

    prob = cp.Problem(cp.Minimize(obj), [x >= 3.14])
    with pytest.raises(AssertionError, match="pass in a recession function"):
        prob.solve()


def test_parameter():
    p = cp.Parameter(nonneg=True)
    x = cp.Variable()
    s = cp.Variable(nonneg=True)
    f = p*cp.square(x)

    obj = cp.perspective(f, s)
    prob = cp.Problem(cp.Minimize(obj), [s <= 1, x >= 2])
    p.value = 99

    prob.solve()

    assert np.isclose(prob.value, 4*p.value)


def test_afine_s():
    # test requiring affine s nonneg
    x = cp.Variable()
    s = cp.Variable(2)
    with pytest.raises(AssertionError, match="s must be a variable"):
        perspective(cp.square(x), cp.sum(s))


def test_dpp():
    x = cp.Variable()
    s = cp.Variable(nonneg=True)
    a = cp.Parameter()

    obj = cp.perspective(cp.square(a+x), s)

    assert not obj.is_dpp()

    obj = cp.perspective(cp.log(a+x), s)

    assert not obj.is_dpp()

def test_s_eq_0():
    # Problem where the optimal s is s = 0
    # Proves that we can support integer / boolean s, where s=0 is more common
    x = cp.Variable(1)
    s = cp.Variable(1, nonneg=True)
    f = x + 1
    f_recession = x
    obj = cp.perspective(f, s, f_recession=f_recession)
    constr = [-cp.square(x) + 1 >= 0]
    
    prob = cp.Problem(cp.Minimize(obj), constr)
    prob.solve()

    assert np.isclose(x.value, -1)
    assert np.isclose(s.value, 0)