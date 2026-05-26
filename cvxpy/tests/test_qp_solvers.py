"""

Copyright 2013 Steven Diamond, 2017 Robin Verschueren

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
import os
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.linalg import lstsq

import cvxpy as cp
import cvxpy.tests.solver_test_helpers as sths
from cvxpy import Maximize, Minimize, Parameter, Problem
from cvxpy.atoms import (
    QuadForm,
    abs,
    huber,
    matrix_frac,
    norm,
    power,
    quad_over_lin,
    sum,
    sum_squares,
)
from cvxpy.error import SolverError
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.flip_objective import FlipObjective
from cvxpy.reductions.solvers.defines import (
    INSTALLED_CONIC_SOLVERS,
    INSTALLED_SOLVERS,
    QP_SOLVERS,
    SOLVER_MAP_CONIC,
    SOLVER_MAP_QP,
)
from cvxpy.reductions.solvers.qp_solvers.osqp_qpif import OSQP
from cvxpy.tests.solver_test_helpers import (
    SolverTestHelper,
    StandardTestInfeasibleProblems,
    StandardTestLPs,
    StandardTestQPs,
)

# --- License checks --- #

def is_mosek_available():
    """Check if MOSEK is installed and a license is available."""
    if 'MOSEK' not in INSTALLED_SOLVERS:
        return False
    try:
        x = cp.Variable()
        cp.Problem(cp.Minimize(x), [x >= 0]).solve(solver=cp.MOSEK)
        return True
    except Exception:
        return False


def is_knitro_available():
    """Check if KNITRO is installed and a license is available.

    Detection is intentionally based on environment variables rather than
    importing ``knitro``: importing it loads the native KNITRO runtime (and
    a bundled OpenMP library on macOS) into the test process, which can
    crash other solvers -- e.g. an IPOPT solve segfaults on macOS once
    knitro has been imported.
    """
    if 'KNITRO' not in INSTALLED_SOLVERS:
        return False
    return bool(
        os.environ.get('ARTELYS_LICENSE')
        or os.environ.get('ARTELYS_LICENSE_NETWORK_ADDR')
    )


def is_xpress_available():
    """Check if XPRESS is installed and a license is available."""
    if 'XPRESS' not in INSTALLED_SOLVERS:
        return False
    try:
        import xpress  # type: ignore
        env = xpress.env()
        status = env.getlicense()
        return status == 0
    except Exception:
        return False


# --- Solver parametrization --- #

def _solver_param(solver):
    """``pytest.param`` for ``solver`` with license-skip and knitro marks."""
    marks = []
    if solver == cp.MOSEK and not is_mosek_available():
        marks.append(pytest.mark.skip(reason='MOSEK license not available'))
    if solver == cp.XPRESS and not is_xpress_available():
        marks.append(pytest.mark.skip(reason='XPRESS license not available'))
    if solver == cp.KNITRO:
        marks.append(pytest.mark.knitro)
        if not is_knitro_available():
            marks.append(pytest.mark.skip(reason='KNITRO not available'))
    return pytest.param(solver, marks=marks, id=solver)


QP_SOLVER_PARAMS = [_solver_param(s) for s in QP_SOLVERS if s in INSTALLED_SOLVERS]

# Conic solvers that support quadratic objectives. KNITRO is excluded -- its
# conic interface with ``use_quad_obj=True`` is unstable in CI.
CONIC_QUAD_OBJ_PARAMS = [
    _solver_param(s)
    for s in INSTALLED_CONIC_SOLVERS
    if s in SOLVER_MAP_CONIC
    and SOLVER_MAP_CONIC[s].supports_quad_obj()
    and s != cp.KNITRO
]


# --- Assertion + solve helpers --- #

def _mat_to_list(mat):
    if isinstance(mat, (np.matrix, np.ndarray)):
        return np.asarray(mat).flatten('F').tolist()
    return mat


def assert_items_almost_equal(a, b, places=5):
    """List-aware almost-equal (matches the old ``BaseTest`` helper)."""
    a = [a] if np.isscalar(a) else _mat_to_list(a)
    b = [b] if np.isscalar(b) else _mat_to_list(b)
    assert len(a) == len(b), f"length mismatch: {len(a)} vs {len(b)}"
    for ai, bi in zip(a, b):
        assert round(ai - bi, places) == 0, f"{ai} != {bi} to {places} places"


def assert_almost_equal(a, b, places=5):
    """Scalar almost-equal (matches ``unittest.TestCase.assertAlmostEqual``)."""
    assert round(a - b, places) == 0, f"{a} != {b} to {places} places"


def check_kkt(problem, places=4):
    """Verify KKT conditions for a solved problem."""
    obj_pair = (problem.objective, None)
    var_pairs = [(v, None) for v in problem.variables()]
    con_pairs = [(c, None) for c in problem.constraints]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    sth.prob = problem
    sth.check_primal_feasibility(places)
    sth.check_complementarity(places)
    sth.check_dual_domains(places)
    sth.check_stationary_lagrangian(places)


def _solve(problem, solver, use_quad_obj=False):
    """Solve ``problem``. With ``use_quad_obj=True`` additionally verify that
    canonicalization introduces no SOC cones -- that's what the QP path
    through a conic solver is supposed to guarantee.
    """
    if use_quad_obj:
        data, _, _ = problem.get_problem_data(
            solver, solver_opts={"use_quad_obj": True}
        )
        assert data["dims"].soc == [], (
            f"Problem should have no SOC cones for QP canonicalization with {solver}"
        )
    kwargs = {"use_quad_obj": True} if use_quad_obj else {}
    return problem.solve(solver=solver, verbose=False, **kwargs)


# --- Fixture --- #

@pytest.fixture
def qp_vars():
    """Fresh per-test bag of Variables (replaces ``QPTestBase.setUp``)."""
    T = 30
    return SimpleNamespace(
        a=Variable(name='a'),
        b=Variable(name='b'),
        c=Variable(name='c'),
        x=Variable(2, name='x'),
        y=Variable(3, name='y'),
        z=Variable(2, name='z'),
        w=Variable(5, name='w'),
        A=Variable((2, 2), name='A'),
        B=Variable((2, 2), name='B'),
        C=Variable((3, 2), name='C'),
        slope=Variable(1, name='slope'),
        offset=Variable(1, name='offset'),
        quadratic_coeff=Variable(1, name='quadratic_coeff'),
        position=Variable((2, T), name='position'),
        velocity=Variable((2, T), name='velocity'),
        force=Variable((2, T - 1), name='force'),
        xs=Variable(80, name='xs'),
        xsr=Variable(50, name='xsr'),
        xef=Variable(80, name='xef'),
    )


# --- QP correctness helpers (one per problem shape) --- #

def _quad_over_lin(v, solver, use_quad_obj=False):
    p = Problem(Minimize(0.5 * quad_over_lin(abs(v.x - 1), 1)), [v.x <= -1])
    _solve(p, solver, use_quad_obj)
    for var in p.variables():
        assert_items_almost_equal(np.array([-1., -1.]), var.value, places=4)
    for con in p.constraints:
        assert_items_almost_equal(np.array([2., 2.]), con.dual_value, places=4)
    check_kkt(p, places=3)


def _abs(v, solver, use_quad_obj=False):
    u = Variable(2)
    prob = Problem(Minimize(sum_squares(u)), [abs(u[1] - u[0]) <= 100])
    assert prob.is_qp()
    result = _solve(prob, solver, use_quad_obj)
    assert_almost_equal(result, 0)


def _power(v, solver, use_quad_obj=False):
    p = Problem(Minimize(sum(power(v.x, 2))), [])
    _solve(p, solver, use_quad_obj)
    for var in p.variables():
        assert_items_almost_equal([0., 0.], var.value, places=4)


def _power_matrix(v, solver, use_quad_obj=False):
    p = Problem(Minimize(sum(power(v.A - 3., 2))), [])
    _solve(p, solver, use_quad_obj)
    for var in p.variables():
        assert_items_almost_equal([3., 3., 3., 3.], var.value, places=4)


def _square_affine(v, solver, use_quad_obj=False):
    A = np.random.randn(10, 2)
    b = np.random.randn(10)
    p = Problem(Minimize(sum_squares(A @ v.x - b)))
    _solve(p, solver, use_quad_obj)
    for var in p.variables():
        assert_items_almost_equal(
            lstsq(A, b)[0].flatten(order='F'), var.value, places=1
        )


def _quad_form(v, solver, use_quad_obj=False):
    np.random.seed(0)
    A = np.random.randn(5, 5)
    z = np.random.randn(5)
    P = A.T.dot(A)
    q = -2 * P.dot(z)
    p = Problem(Minimize(QuadForm(v.w, P) + q.T @ v.w))
    _solve(p, solver, use_quad_obj)
    for var in p.variables():
        assert_items_almost_equal(z, var.value, places=4)


def _rep_quad_form(v, solver, use_quad_obj=False):
    """A problem where the quad_form term is used multiple times."""
    np.random.seed(0)
    A = np.random.randn(5, 5)
    z = np.random.randn(5)
    P = A.T.dot(A)
    q = -2 * P.dot(z)
    qf = QuadForm(v.w, P)
    p = Problem(Minimize(0.5 * qf + 0.5 * qf + q.T @ v.w))
    _solve(p, solver, use_quad_obj)
    for var in p.variables():
        assert_items_almost_equal(z, var.value, places=4)


def _affine_problem(v, solver, use_quad_obj=False):
    np.random.seed(0)
    A = np.maximum(np.random.randn(5, 2), 0)
    b = np.maximum(np.random.randn(5), 0)
    p = Problem(Minimize(sum(v.x)), [v.x >= 0, A @ v.x <= b])
    _solve(p, solver, use_quad_obj)
    for var in p.variables():
        assert_items_almost_equal([0., 0.], var.value, places=3)
    check_kkt(p, places=3)


def _maximize_problem(v, solver, use_quad_obj=False):
    np.random.seed(0)
    A = np.maximum(np.random.randn(5, 2), 0)
    b = np.maximum(np.random.randn(5), 0)
    p = Problem(Maximize(-sum(v.x)), [v.x >= 0, A @ v.x <= b])
    _solve(p, solver, use_quad_obj)
    for var in p.variables():
        assert_items_almost_equal([0., 0.], var.value, places=3)
    check_kkt(p, places=3)


def _quad_form_coeff(v, solver, use_quad_obj=False):
    np.random.seed(0)
    A = np.random.randn(5, 5)
    z = np.random.randn(5)
    P = A.T.dot(A)
    q = -2 * P.dot(z)
    p = Problem(Minimize(QuadForm(v.w, P) + q.T @ v.w))
    _solve(p, solver, use_quad_obj)
    for var in p.variables():
        assert_items_almost_equal(z, var.value, places=4)


def _quad_form_bound(v, solver, use_quad_obj=False):
    P = np.array([[13, 12, -2], [12, 17, 6], [-2, 6, 12]])
    q = np.array([[-22], [-14.5], [13]])
    y_star = np.array([[1], [0.5], [-1]])
    p = Problem(Minimize(0.5 * QuadForm(v.y, P) + q.T @ v.y + 1),
                [v.y >= -1, v.y <= 1])
    _solve(p, solver, use_quad_obj)
    for var in p.variables():
        assert_items_almost_equal(y_star, var.value, places=4)
    check_kkt(p)


def _regression_1(v, solver, use_quad_obj=False):
    np.random.seed(1)
    n = 100
    true_coeffs = np.array([[2, -2, 0.5]]).T
    x_data = np.atleast_2d(np.random.rand(n) * 5)
    x_data_expanded = np.atleast_2d(
        np.vstack([np.power(x_data, i) for i in range(1, 4)])
    )
    y_data = np.atleast_2d(
        x_data_expanded.T.dot(true_coeffs) + 0.5 * np.random.rand(n, 1)
    )
    line = v.offset + x_data * v.slope
    fit_error = sum_squares(line.T - y_data)
    p = Problem(Minimize(fit_error), [])
    _solve(p, solver, use_quad_obj)
    assert_almost_equal(1171.60037715, p.value, places=4)


def _regression_2(v, solver, use_quad_obj=False):
    np.random.seed(1)
    n = 100
    true_coeffs = np.array([2, -2, 0.5])
    x_data = np.random.rand(n) * 5
    x_data_expanded = np.vstack([np.power(x_data, i) for i in range(1, 4)])
    y_data = x_data_expanded.T.dot(true_coeffs) + 0.5 * np.random.rand(n)
    quadratic = (
        v.offset + x_data * v.slope + v.quadratic_coeff * np.power(x_data, 2)
    )
    fit_error = sum_squares(quadratic.T - y_data)
    p = Problem(Minimize(fit_error), [])
    _solve(p, solver, use_quad_obj)
    assert_almost_equal(139.225660756, p.value, places=4)


def _control(v, solver, use_quad_obj=False):
    initial_velocity = np.array([-20, 100])
    final_position = np.array([100, 100])
    T = 30
    h = 0.1
    mass = 1
    drag = 0.1
    g = np.array([0, -9.8])
    constraints = []
    for i in range(T - 1):
        constraints += [v.position[:, i + 1] == v.position[:, i] + h * v.velocity[:, i]]
        acceleration = v.force[:, i] / mass + g - drag * v.velocity[:, i]
        constraints += [v.velocity[:, i + 1] == v.velocity[:, i] + h * acceleration]
    constraints += [v.position[:, 0] == 0]
    constraints += [v.position[:, -1] == final_position]
    constraints += [v.velocity[:, 0] == initial_velocity]
    constraints += [v.velocity[:, -1] == 0]
    p = Problem(Minimize(.01 * sum_squares(v.force)), constraints)
    _solve(p, solver, use_quad_obj)
    assert_almost_equal(1059.616, p.value, places=1)
    # KKT check skipped: check_stationary_lagrangian fails for 2D matrix
    # variables due to inconsistent gradient ordering (sum_squares uses
    # C order, constraint terms use F order). TODO fix this


def _sparse_system(v, solver, use_quad_obj=False):
    m, n = 100, 80
    np.random.seed(1)
    A = sp.random_array((m, n), density=0.4)
    b = np.random.randn(m)
    p = Problem(Minimize(sum_squares(A @ v.xs - b)), [v.xs == 0])
    _solve(p, solver, use_quad_obj)
    assert_almost_equal(b.T.dot(b), p.value, places=4)


def _smooth_ridge(v, solver, use_quad_obj=False):
    np.random.seed(1)
    n = 50
    k = 20
    A = np.ones((k, n))
    b = np.ones(k)
    obj = sum_squares(A @ v.xsr - b) + sum_squares(v.xsr[:-1] - v.xsr[1:])
    p = Problem(Minimize(obj), [])
    _solve(p, solver, use_quad_obj)
    assert_almost_equal(0, p.value, places=4)


def _huber_small(v, solver, use_quad_obj=False, places=4):
    x = Variable(3)
    objective = sum(huber(x))
    p = Problem(Minimize(objective), [x[2] >= 3])
    _solve(p, solver, use_quad_obj)
    assert_almost_equal(3, x.value[2], places=places)
    assert_almost_equal(5, objective.value, places=places)
    check_kkt(p, places=places)


def _huber(v, solver, use_quad_obj=False):
    n, m = 3, 5
    data = [0.89, 0.39, 0.96, 0.34, 0.68, 0.18,
            0.63, 0.42, 0.51, 0.66, 0.43, 0.77]
    indices = [0, 1, 2, 3, 4, 2, 3, 0, 1, 2, 3, 4]
    indptr = [0, 5, 7, 12]
    A = sp.csc_array((data, indices, indptr), shape=(m, n))
    x_true = np.random.randn(n) / np.sqrt(n)
    ind95 = (np.random.rand(m) < 0.95).astype(float)
    b = A.dot(x_true) + np.multiply(0.5 * np.random.randn(m), ind95) \
        + np.multiply(10. * np.random.rand(m), 1. - ind95)
    x = Variable(n)
    objective = sum(huber(A @ x - b))
    p = Problem(Minimize(objective))
    _solve(p, solver, use_quad_obj)
    assert_almost_equal(1.452797819667, objective.value, places=3)
    assert_items_almost_equal(
        x.value, [1.20524645, -0.85271489, -0.50838494], places=3
    )


def _equivalent_forms_setup(seed=1):
    np.random.seed(seed)
    m, n, r = 100, 80, 70
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    G = np.random.randn(r, n)
    h = np.random.randn(r)
    return A, b, G, h


def _equivalent_forms_1(v, solver, use_quad_obj=False):
    A, b, G, h = _equivalent_forms_setup()
    p = Problem(Minimize(.1 * sum((A @ v.xef - b) ** 2)), [G @ v.xef == h])
    _solve(p, solver, use_quad_obj)
    assert_almost_equal(p.value, 68.1119420108, places=4)
    check_kkt(p, places=4)


def _equivalent_forms_2(v, solver, use_quad_obj=False):
    A, b, G, h = _equivalent_forms_setup()
    # ||Ax-b||^2 = x^T (A^T A) x - 2(A^T b)^T x + ||b||^2
    P = A.T @ A
    q = -2 * A.T @ b
    r = b.T @ b
    obj = .1 * (QuadForm(v.xef, P) + q.T @ v.xef + r)
    p = Problem(Minimize(obj), [G @ v.xef == h])
    _solve(p, solver, use_quad_obj)
    assert_almost_equal(p.value, 68.1119420108, places=4)
    check_kkt(p, places=4)


def _equivalent_forms_3(v, solver, use_quad_obj=False):
    A, b, G, h = _equivalent_forms_setup()
    P = A.T @ A
    q = -2 * A.T @ b
    r = b.T @ b
    Pinv = np.linalg.inv(P)
    obj = .1 * (matrix_frac(v.xef, Pinv) + q.T @ v.xef + r)
    p = Problem(Minimize(obj), [G @ v.xef == h])
    _solve(p, solver, use_quad_obj)
    assert_almost_equal(p.value, 68.1119420108, places=4)
    check_kkt(p, places=4)


# --- Per-solver tests --- #

@pytest.mark.parametrize("solver", QP_SOLVER_PARAMS)
def test_qp_correctness(qp_vars, solver):
    """Run all QP correctness helpers against one native QP solver."""
    _quad_over_lin(qp_vars, solver)
    _power(qp_vars, solver)
    _power_matrix(qp_vars, solver)
    _square_affine(qp_vars, solver)
    _quad_form(qp_vars, solver)
    _affine_problem(qp_vars, solver)
    _maximize_problem(qp_vars, solver)
    _abs(qp_vars, solver)
    _quad_form_coeff(qp_vars, solver)
    _quad_form_bound(qp_vars, solver)
    _regression_1(qp_vars, solver)
    _regression_2(qp_vars, solver)
    _rep_quad_form(qp_vars, solver)
    _control(qp_vars, solver)
    _sparse_system(qp_vars, solver)
    _smooth_ridge(qp_vars, solver)
    _huber_small(qp_vars, solver)
    _huber(qp_vars, solver)
    _equivalent_forms_1(qp_vars, solver)
    _equivalent_forms_2(qp_vars, solver)
    if solver != cp.KNITRO:
        # KNITRO does not support matrix_frac.
        _equivalent_forms_3(qp_vars, solver)


@pytest.mark.parametrize("solver", CONIC_QUAD_OBJ_PARAMS)
def test_qp_conic_quad_obj(qp_vars, solver):
    """Run QP correctness helpers via conic solvers with use_quad_obj=True.

    Solvers with ``REQUIRES_CONSTR=True`` skip helpers whose problems are
    unconstrained -- those canonicalize to m=0 and are rejected.
    """
    requires_constr = SOLVER_MAP_CONIC[solver].REQUIRES_CONSTR
    _quad_over_lin(qp_vars, solver, use_quad_obj=True)
    if not requires_constr:
        _power(qp_vars, solver, use_quad_obj=True)
    _power_matrix(qp_vars, solver, use_quad_obj=True)
    _square_affine(qp_vars, solver, use_quad_obj=True)
    if not requires_constr:
        _quad_form(qp_vars, solver, use_quad_obj=True)
    _affine_problem(qp_vars, solver, use_quad_obj=True)
    _maximize_problem(qp_vars, solver, use_quad_obj=True)
    _abs(qp_vars, solver, use_quad_obj=True)
    if not requires_constr:
        _quad_form_coeff(qp_vars, solver, use_quad_obj=True)
    _quad_form_bound(qp_vars, solver, use_quad_obj=True)
    _regression_1(qp_vars, solver, use_quad_obj=True)
    _regression_2(qp_vars, solver, use_quad_obj=True)
    if not requires_constr:
        _rep_quad_form(qp_vars, solver, use_quad_obj=True)
    _control(qp_vars, solver, use_quad_obj=True)
    _sparse_system(qp_vars, solver, use_quad_obj=True)
    _smooth_ridge(qp_vars, solver, use_quad_obj=True)
    _huber_small(qp_vars, solver, use_quad_obj=True, places=3)
    _huber(qp_vars, solver, use_quad_obj=True)
    _equivalent_forms_1(qp_vars, solver, use_quad_obj=True)
    _equivalent_forms_2(qp_vars, solver, use_quad_obj=True)
    _equivalent_forms_3(qp_vars, solver, use_quad_obj=True)


@pytest.mark.parametrize("solver", QP_SOLVER_PARAMS)
def test_qp_bound_attr(solver):
    solver_cls = SOLVER_MAP_QP.get(solver)
    if solver_cls is None or not getattr(solver_cls, "BOUNDED_VARIABLES", False):
        pytest.skip(f"{solver} does not support bounded-variable attributes")
    StandardTestQPs.test_qp_bound_attr(solver=solver)


@pytest.mark.parametrize("solver", QP_SOLVER_PARAMS)
def test_parametric(solver):
    """Solve parametric problem vs full problem."""
    x = Variable()
    a = 10
    b_vec = [-10, -2.]

    x_full, obj_full = [], []
    for b in b_vec:
        prob = Problem(Minimize(a * (x ** 2) + b * x), [0 <= x, x <= 1])
        prob.solve(solver=solver)
        x_full.append(x.value)
        obj_full.append(prob.value)

    x_param, obj_param = [], []
    b = Parameter()
    prob = Problem(Minimize(a * (x ** 2) + b * x), [0 <= x, x <= 1])
    for b_value in b_vec:
        b.value = b_value
        prob.solve(solver=solver)
        x_param.append(x.value)
        obj_param.append(prob.value)

    for i in range(len(b_vec)):
        assert_items_almost_equal(x_full[i], x_param[i], places=3)
        assert_almost_equal(obj_full[i], obj_param[i])


# --- Solver-specific tests --- #

def test_warm_start():
    m, n = 200, 100
    np.random.seed(1)
    A = np.random.randn(m, n)
    b = Parameter(m)
    x = Variable(n)
    prob = Problem(Minimize(sum_squares(A @ x - b)))

    b.value = np.random.randn(m)
    result = prob.solve(solver="OSQP", warm_start=False)
    result2 = prob.solve(solver="OSQP", warm_start=True)
    assert_almost_equal(result, result2)
    b.value = np.random.randn(m)
    result = prob.solve(solver="OSQP", warm_start=True)
    result2 = prob.solve(solver="OSQP", warm_start=False)
    assert_almost_equal(result, result2)


@pytest.mark.skipif(cp.QPALM not in INSTALLED_SOLVERS, reason="QPALM is not installed")
def test_qpalm_warmstart():
    m, n = 200, 100
    np.random.seed(1)
    A = np.random.randn(m, n)
    b = Parameter(m)
    x = Variable(n)
    prob = Problem(Minimize(sum_squares(A @ x - b)))

    b.value = np.random.randn(m)
    result = prob.solve(solver=cp.QPALM, warm_start=False)
    result2 = prob.solve(solver=cp.QPALM, warm_start=True)
    assert_almost_equal(result, result2)
    b.value = np.random.randn(m)
    result = prob.solve(solver=cp.QPALM, warm_start=True)
    result2 = prob.solve(solver=cp.QPALM, warm_start=False)
    assert_almost_equal(result, result2)


@pytest.mark.skipif(cp.GUROBI not in INSTALLED_SOLVERS, reason="GUROBI is not installed")
def test_gurobi_warmstart():
    """Test Gurobi warm start with a user provided point."""
    import gurobipy
    m, n = 4, 3
    y = Variable(nonneg=True)
    X = Variable((m, n))
    X_vals = np.reshape(np.arange(m * n), (m, n))
    prob = Problem(Minimize(y ** 2 + cp.sum(X)), [X == X_vals])
    X.value = X_vals + 1
    prob.solve(solver=cp.GUROBI, warm_start=True)
    model = prob.solver_stats.extra_stats
    model_x = model.getVars()
    assert gurobipy.GRB.UNDEFINED == model_x[0].start
    assert np.isclose(0, model_x[0].x)
    for i in range(1, X.size + 1):
        row = (i - 1) % X.shape[0]
        col = (i - 1) // X.shape[0]
        assert X_vals[row, col] + 1 == model_x[i].start
        assert np.isclose(X.value[row, col], model_x[i].x)


@pytest.mark.skipif(cp.XPRESS not in INSTALLED_SOLVERS, reason="XPRESS is not installed")
def test_xpress_warmstart():
    """Test XPRESS warm start with a user provided point."""
    m, n = 20, 10
    np.random.seed(1)
    A = np.random.randn(m, n)
    b = Parameter(m)
    x = Variable(n, integer=True)
    prob = Problem(Minimize(sum_squares(A @ x - b)))

    b.value = np.random.randn(m)
    result = prob.solve(solver=cp.XPRESS, warm_start=False)
    result2 = prob.solve(solver=cp.XPRESS, warm_start=True)
    assert_almost_equal(result, result2)
    x.value = x.value.astype(np.int64)

    xprime = Variable(n, integer=True)
    prob = Problem(Minimize(sum_squares(A @ xprime - b)))
    xprime.value = x.value
    result = prob.solve(solver=cp.XPRESS, warm_start=True)
    result2 = prob.solve(solver=cp.XPRESS, warm_start=False)
    assert_almost_equal(result, result2)


@pytest.mark.skipif(cp.HIGHS not in INSTALLED_SOLVERS, reason="HIGHS is not installed")
def test_highs_warmstart():
    m, n = 200, 100
    np.random.seed(1)
    A = np.random.randn(m, n)
    b = Parameter(m)
    x = Variable(n)
    prob = Problem(Minimize(sum_squares(A @ x - b)))

    b.value = np.random.randn(m)
    result = prob.solve(solver=cp.HIGHS, warm_start=False)
    result2 = prob.solve(solver=cp.HIGHS, warm_start=True)
    assert_almost_equal(result, result2)
    b.value = np.random.randn(m)
    result = prob.solve(solver=cp.HIGHS, warm_start=True)
    result2 = prob.solve(solver=cp.HIGHS, warm_start=False)
    assert_almost_equal(result, result2)


@pytest.mark.skipif(cp.HIGHS not in INSTALLED_SOLVERS, reason="HIGHS is not installed")
def test_highs_cvar():
    """CVaR constraint regression for https://github.com/cvxpy/cvxpy/issues/2836."""
    num_stocks, num_samples = 5, 25
    np.random.seed(1)
    pnl_samples = np.random.uniform(low=0.0, high=1.0, size=(num_samples, num_stocks))
    pnl_expected = pnl_samples.mean(axis=0)

    quantile = 0.05
    w = cp.Variable(num_stocks, nonneg=True)
    cvar = cp.cvar(pnl_samples @ w, 1 - quantile)
    problem = cp.Problem(cp.Maximize(w @ pnl_expected), [cvar <= 0.5])
    problem.solve(solver=cp.HIGHS)
    assert problem.status == cp.OPTIMAL


@pytest.mark.skipif(cp.PIQP not in INSTALLED_SOLVERS, reason="PIQP is not installed")
def test_piqp_warmstart():
    m, n = 200, 100
    np.random.seed(1)
    A = np.random.randn(m, n)
    b = Parameter(m)
    x = Variable(n)
    prob = Problem(Minimize(sum_squares(A @ x - b)))

    b.value = np.random.randn(m)
    result = prob.solve(solver=cp.PIQP, warm_start=False)
    result2 = prob.solve(solver=cp.PIQP, warm_start=True)
    assert_almost_equal(result, result2)
    b.value = np.random.randn(m)
    result = prob.solve(solver=cp.PIQP, warm_start=True)
    result2 = prob.solve(solver=cp.PIQP, warm_start=False)
    assert_almost_equal(result, result2)


@pytest.mark.skipif(cp.COPT not in INSTALLED_SOLVERS, reason="COPT is not installed")
def test_copt_warmstart():
    m, n = 200, 100
    np.random.seed(1)
    A = np.random.randn(m, n)
    b = Parameter(m)
    x = Variable(n)
    prob = Problem(Minimize(sum_squares(A @ x - b)))

    b.value = np.random.randn(m)
    result = prob.solve(solver=cp.COPT, warm_start=False)
    result2 = prob.solve(solver=cp.COPT, warm_start=True)
    assert_almost_equal(result, result2)
    b.value = np.random.randn(m)
    result = prob.solve(solver=cp.COPT, warm_start=True)
    result2 = prob.solve(solver=cp.COPT, warm_start=False)
    assert_almost_equal(result, result2)


def test_square_param():
    """Issue arising with square plus parameter."""
    a = Parameter(value=1)
    b = Variable()
    prob = Problem(Minimize(b ** 2 + abs(a)))
    prob.solve(solver="SCS")
    assert_almost_equal(prob.value, 1.0)


def test_gurobi_time_limit_no_solution(qp_vars):
    """If Gurobi hits its time limit before finding a solution, the solve
    must return cleanly and expose solver stats.
    """
    if cp.GUROBI in INSTALLED_SOLVERS:
        import gurobipy
        prob = Problem(Minimize(qp_vars.x[0]), [qp_vars.x[0] >= 1])
        try:
            prob.solve(solver=cp.GUROBI, TimeLimit=0.0)
        except Exception as e:
            pytest.fail(f"An exception {e} is raised instead of returning a result.")

        extra_stats = getattr(getattr(prob, "solver_stats", None), "extra_stats", None)
        assert extra_stats, "Solver stats have not been returned."

        if getattr(extra_stats, "SolCount", None):
            pytest.skip("Gurobi has found a solution, the test is not relevant anymore.")

        if getattr(extra_stats, "Status", None) != gurobipy.GRB.TIME_LIMIT:
            pytest.skip(
                "Gurobi terminated for a different reason than reaching time limit, "
                "the test is not relevant anymore."
            )
    else:
        prob = Problem(Minimize(norm(qp_vars.x, 1)), [qp_vars.x == 0])
        with pytest.raises(Exception) as exc_info:
            prob.solve(solver=cp.GUROBI, TimeLimit=0)
        assert str(exc_info.value) == f"The solver {cp.GUROBI} is not installed."


def test_gurobi_environment(qp_vars):
    """Gurobi environments (with licensing/model parameter data) can be passed
    through to the underlying Model.
    """
    if cp.GUROBI in INSTALLED_SOLVERS:
        import gurobipy

        params = {
            'MIPGap': np.random.random(),
            'AggFill': np.random.randint(10),
            'PerturbValue': np.random.random(),
        }
        custom_env = gurobipy.Env()
        for k, v in params.items():
            custom_env.setParam(k, v)

        sth = StandardTestLPs.test_lp_0(solver='GUROBI', env=custom_env)
        model = sth.prob.solver_stats.extra_stats
        for k, v in params.items():
            _, _, p_val, _, _, _ = model.getParamInfo(k)
            assert v == p_val
    else:
        prob = Problem(Minimize(norm(qp_vars.x, 1)), [qp_vars.x == 0])
        with pytest.raises(Exception) as exc_info:
            prob.solve(solver=cp.GUROBI, TimeLimit=0)
        assert str(exc_info.value) == f"The solver {cp.GUROBI} is not installed."


def test_osqp_infeasible_lp_ineq_constraints():
    StandardTestInfeasibleProblems.test_lp_ineq_constraints(solver=cp.OSQP)


def test_osqp_infeasible_lp_eq_constraints():
    StandardTestInfeasibleProblems.test_lp_eq_constraints(solver=cp.OSQP)


def test_highs_infeasible_lp_ineq_constraints():
    StandardTestInfeasibleProblems.test_lp_ineq_constraints(solver=cp.HIGHS)


def test_highs_infeasible_lp_eq_constraints():
    StandardTestInfeasibleProblems.test_lp_eq_constraints(solver=cp.HIGHS)


@pytest.mark.skipif(cp.HIGHS not in INSTALLED_SOLVERS, reason="HIGHS is not installed")
def test_highs_dense_quad_form():
    """Regression test for https://github.com/cvxpy/cvxpy/issues/3301.

    A dense quad_form applied to a linear expression (not a raw Variable)
    produces a Hessian whose upper and lower triangles may differ by
    floating-point epsilon after canonicalization. Passing such a Hessian
    in square format to HiGHS >= 1.14.0 triggers an asymmetry error
    and native heap corruption. Using triangular format avoids this.
    """
    rng = np.random.default_rng(42)
    n_vars, n_nodes = 60, 20

    rows, cols, vals = [], [], []
    for i in range(n_vars):
        j, k = rng.choice(n_nodes, size=2, replace=False)
        rows += [i, i]
        cols += [j, k]
        vals += [1.0, -1.0]
    M = sp.csr_matrix((vals, (rows, cols)), shape=(n_vars, n_nodes))

    A = rng.standard_normal((n_nodes, n_nodes))
    raw = A @ A.T + rng.standard_normal((n_nodes, n_nodes)) * 0.01
    sym = 0.5 * (raw + raw.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    Sigma = eigvecs @ np.diag(np.maximum(eigvals, 0.0)) @ eigvecs.T

    x = cp.Variable(n_vars, nonneg=True)
    prob = cp.Problem(
        cp.Maximize(cp.sum(x) - 0.1 * cp.quad_form(x @ M, Sigma)),
        [x <= 10],
    )
    prob.solve(solver=cp.HIGHS)
    assert prob.status == cp.OPTIMAL


# --- MPAX tests --- #

mpax_skip = pytest.mark.skipif(
    'MPAX' not in INSTALLED_SOLVERS, reason='MPAX is not installed.'
)


@mpax_skip
def test_mpax_lp_0():
    StandardTestLPs.test_lp_0(solver='MPAX')


@mpax_skip
def test_mpax_lp_1():
    StandardTestLPs.test_lp_1(solver='MPAX')


@mpax_skip
def test_mpax_lp_2():
    StandardTestLPs.test_lp_2(solver='MPAX')


@mpax_skip
def test_mpax_lp_3():
    sth = sths.lp_3()
    with pytest.warns(Warning):
        sth.prob.solve(solver='MPAX')
        assert sth.prob.status == cp.settings.INFEASIBLE_OR_UNBOUNDED


@mpax_skip
def test_mpax_lp_4():
    sth = sths.lp_4()
    with pytest.warns(Warning):
        sth.prob.solve(solver='MPAX')
        assert sth.prob.status == cp.settings.INFEASIBLE_OR_UNBOUNDED


@mpax_skip
def test_mpax_lp_5():
    StandardTestLPs.test_lp_5(solver='MPAX')


@mpax_skip
def test_mpax_lp_6():
    StandardTestLPs.test_lp_6(solver='MPAX')


@mpax_skip
def test_mpax_warmstart():
    x = cp.Variable(shape=(2,), name='x')
    prob = cp.Problem(
        cp.Minimize(-4 * x[0] - 5 * x[1]),
        [2 * x[0] + x[1] <= 3, x[0] + 2 * x[1] <= 3, x[0] >= 0, x[1] >= 0],
    )
    assert_almost_equal(prob.solve(solver='MPAX', warm_start=False), -9, places=4)
    assert_almost_equal(prob.solve(solver='MPAX', warm_start=True), -9, places=4)


@mpax_skip
def test_mpax_qp_0():
    StandardTestQPs.test_qp_0(solver='MPAX')


# --- QP solver cone-type validation --- #

def _apply_reductions(problem):
    """Apply the full reduction chain to get a ParamConeProg."""
    reductions = []
    if isinstance(problem.objective, cp.Maximize):
        reductions.append(FlipObjective())
    reductions.extend([Dcp2Cone(), CvxAttr2Constr(), ConeMatrixStuffing()])
    reduced = problem
    for reduction in reductions:
        reduced = reduction.apply(reduced)[0]
    return reduced


def test_qp_solver_rejects_exponential_cones():
    """QP solver rejects problems with exponential cones."""
    x = cp.Variable()
    prob = cp.Problem(cp.Maximize(cp.log(x)), [x <= 1])
    osqp_solver = OSQP()
    assert not osqp_solver.accepts(_apply_reductions(prob))
    with pytest.raises(SolverError) as exc_info:
        osqp_solver.apply(_apply_reductions(prob))
    assert "exponential cones" in str(exc_info.value)
    assert "OSQP" in str(exc_info.value)


def test_qp_solver_rejects_psd_cones():
    """QP solver rejects problems with PSD cones."""
    X = cp.Variable((2, 2), symmetric=True)
    prob = cp.Problem(cp.Minimize(cp.trace(X)), [X >> 0, X[0, 0] >= 1])
    osqp_solver = OSQP()
    assert not osqp_solver.accepts(_apply_reductions(prob))
    with pytest.raises(SolverError) as exc_info:
        osqp_solver.apply(_apply_reductions(prob))
    assert "PSD cones" in str(exc_info.value)


def test_qp_solver_rejects_soc_cones():
    """QP solver rejects problems with second-order cones."""
    x = cp.Variable(3)
    prob = cp.Problem(cp.Minimize(cp.norm(x)), [cp.sum(x) == 1])
    osqp_solver = OSQP()
    assert not osqp_solver.accepts(_apply_reductions(prob))
    with pytest.raises(SolverError) as exc_info:
        osqp_solver.apply(_apply_reductions(prob))
    assert "second-order cones" in str(exc_info.value)
