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
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.linalg import lstsq

import cvxpy as cp
import cvxpy.tests.solver_test_helpers as sths
from cvxpy.error import SolverError
from cvxpy.reductions import (
    ConeMatrixStuffing,
    CvxAttr2Constr,
    Dcp2Cone,
    FlipObjective,
)
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
from cvxpy.tests.test_conic_solvers import is_knitro_available, is_mosek_available

# --- License checks --- #

# is_mosek_available / is_knitro_available come from test_conic_solvers so the
# canonical, KNITRO-OpenMP-safe implementations live in one place.

def is_xpress_available():
    """Check if XPRESS is installed and a usable license can be acquired.

    The Community license bundled with the ``xpress`` package is sufficient for
    the small problems in this suite, so we only confirm that a problem object
    can be created -- that is where XPRESS acquires its license. The previous
    ``xpress.env().getlicense()`` API was removed in modern xpress, so the check
    raised ``AttributeError`` and always returned False, silently skipping every
    license-gated XPRESS QP test.
    """
    if 'XPRESS' not in INSTALLED_SOLVERS:
        return False
    try:
        import xpress  # type: ignore
        xpress.problem()
        return True
    except Exception:
        return False


# --- Solver parametrization --- #

def _solver_marks(solver):
    """License-skip and ``knitro`` marks for a single solver."""
    marks = []
    if solver == cp.MOSEK and not is_mosek_available():
        marks.append(pytest.mark.skip(reason='MOSEK license not available'))
    if solver == cp.XPRESS and not is_xpress_available():
        marks.append(pytest.mark.skip(reason='XPRESS license not available'))
    if solver == cp.KNITRO:
        marks.append(pytest.mark.knitro)
        if not is_knitro_available():
            marks.append(pytest.mark.skip(reason='KNITRO not available'))
    return marks


# Native QP solvers, used by tests that only run against the QP path
# (warm-start tests, ``test_qp_bound_attr``, ``test_parametric``).
QP_SOLVER_PARAMS = [
    pytest.param(s, marks=_solver_marks(s), id=s)
    for s in QP_SOLVERS if s in INSTALLED_SOLVERS
]

# Every (solver, use_quad_obj) variant we want to run each QP correctness test
# against: every native QP solver with use_quad_obj=False, plus every conic
# solver that supports quadratic objectives with use_quad_obj=True. KNITRO is
# excluded from the conic side -- its conic interface with use_quad_obj=True
# is unstable in CI.
QP_PARAMS = (
    [pytest.param(s, False, marks=_solver_marks(s), id=s)
     for s in QP_SOLVERS if s in INSTALLED_SOLVERS]
    + [pytest.param(s, True, marks=_solver_marks(s), id=f"{s}-quadobj")
       for s in INSTALLED_CONIC_SOLVERS
       if s in SOLVER_MAP_CONIC
       and SOLVER_MAP_CONIC[s].supports_quad_obj()
       and s != cp.KNITRO]
)


# --- Solve + KKT helpers --- #

def _solve(problem, solver, use_quad_obj):
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
    try:
        return problem.solve(solver=solver, verbose=False, **kwargs)
    except Exception as exc:
        _skip_if_xpress_community_limit(solver, exc)
        raise


def _skip_if_xpress_community_limit(solver, exc):
    """Skip when a problem exceeds the XPRESS Community license size limit.

    The Community license bundled with the ``xpress`` package caps problems at
    200 rows+columns; larger problems raise a size-limit ``SolverError``. Skip
    those rather than fail -- they run fully under a full XPRESS license, and
    the smaller problems still exercise the QP interface.
    """
    if solver == cp.XPRESS and "too many rows and columns" in str(exc):
        pytest.skip("XPRESS Community license problem-size limit (200 rows+cols)")


def check_kkt(problem, places=4):
    """Verify KKT conditions for a solved problem.

    Stays in ``places`` units because ``SolverTestHelper`` exposes its
    tolerances as decimal places, not absolute tolerances.
    """
    obj_pair = (problem.objective, None)
    var_pairs = [(v, None) for v in problem.variables()]
    con_pairs = [(c, None) for c in problem.constraints]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    sth.prob = problem
    sth.check_primal_feasibility(places)
    sth.check_complementarity(places)
    sth.check_dual_domains(places)
    sth.check_stationary_lagrangian(places)


def _skip_if_requires_constr(solver, use_quad_obj):
    """Conic solvers with ``REQUIRES_CONSTR=True`` reject unconstrained
    problems (those canonicalize to m=0). Skip such ``(solver, helper)``
    combinations when running the conic + use_quad_obj path.
    """
    if use_quad_obj and SOLVER_MAP_CONIC[solver].REQUIRES_CONSTR:
        pytest.skip(f"{solver} requires constraints; this problem is unconstrained")


# --- QP correctness tests --- #

@pytest.mark.parametrize("solver,use_quad_obj", QP_PARAMS)
def test_quad_over_lin(solver, use_quad_obj):
    x = cp.Variable(2)
    p = cp.Problem(cp.Minimize(0.5 * cp.quad_over_lin(cp.abs(x - 1), 1)), [x <= -1])
    _solve(p, solver, use_quad_obj)
    np.testing.assert_allclose(x.value, [-1., -1.], atol=1e-4)
    np.testing.assert_allclose(p.constraints[0].dual_value, [2., 2.], atol=1e-4)
    check_kkt(p, places=3)


@pytest.mark.parametrize("solver,use_quad_obj", QP_PARAMS)
def test_abs(solver, use_quad_obj):
    u = cp.Variable(2)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(u)), [cp.abs(u[1] - u[0]) <= 100])
    assert prob.is_qp()
    result = _solve(prob, solver, use_quad_obj)
    np.testing.assert_allclose(result, 0, atol=1e-5)


@pytest.mark.parametrize("solver,use_quad_obj", QP_PARAMS)
def test_power(solver, use_quad_obj):
    _skip_if_requires_constr(solver, use_quad_obj)
    x = cp.Variable(2)
    p = cp.Problem(cp.Minimize(cp.sum(cp.power(x, 2))), [])
    _solve(p, solver, use_quad_obj)
    np.testing.assert_allclose(x.value, [0., 0.], atol=1e-4)


@pytest.mark.parametrize("solver,use_quad_obj", QP_PARAMS)
def test_power_matrix(solver, use_quad_obj):
    X = cp.Variable((2, 2))
    p = cp.Problem(cp.Minimize(cp.sum(cp.power(X - 3., 2))), [])
    _solve(p, solver, use_quad_obj)
    np.testing.assert_allclose(X.value, np.full((2, 2), 3.), atol=1e-4)


@pytest.mark.parametrize("solver,use_quad_obj", QP_PARAMS)
def test_square_affine(solver, use_quad_obj):
    x = cp.Variable(2)
    A = np.random.randn(10, 2)
    b = np.random.randn(10)
    p = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))
    _solve(p, solver, use_quad_obj)
    np.testing.assert_allclose(x.value, lstsq(A, b)[0], atol=1e-1)


@pytest.mark.parametrize("solver,use_quad_obj", QP_PARAMS)
def test_quad_form(solver, use_quad_obj):
    _skip_if_requires_constr(solver, use_quad_obj)
    np.random.seed(0)
    A = np.random.randn(5, 5)
    z = np.random.randn(5)
    P = A.T @ A
    q = -2 * P @ z
    w = cp.Variable(5)
    p = cp.Problem(cp.Minimize(cp.QuadForm(w, P) + q.T @ w))
    _solve(p, solver, use_quad_obj)
    np.testing.assert_allclose(w.value, z, atol=1e-4)


@pytest.mark.parametrize("solver,use_quad_obj", QP_PARAMS)
def test_rep_quad_form(solver, use_quad_obj):
    """A problem where the quad_form term is used multiple times."""
    _skip_if_requires_constr(solver, use_quad_obj)
    np.random.seed(0)
    A = np.random.randn(5, 5)
    z = np.random.randn(5)
    P = A.T @ A
    q = -2 * P @ z
    w = cp.Variable(5)
    qf = cp.QuadForm(w, P)
    p = cp.Problem(cp.Minimize(0.5 * qf + 0.5 * qf + q.T @ w))
    _solve(p, solver, use_quad_obj)
    np.testing.assert_allclose(w.value, z, atol=1e-4)


@pytest.mark.parametrize("solver,use_quad_obj", QP_PARAMS)
def test_affine_problem(solver, use_quad_obj):
    np.random.seed(0)
    A = np.maximum(np.random.randn(5, 2), 0)
    b = np.maximum(np.random.randn(5), 0)
    x = cp.Variable(2)
    p = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0, A @ x <= b])
    _solve(p, solver, use_quad_obj)
    np.testing.assert_allclose(x.value, [0., 0.], atol=1e-3)
    check_kkt(p, places=3)


@pytest.mark.parametrize("solver,use_quad_obj", QP_PARAMS)
def test_maximize_problem(solver, use_quad_obj):
    np.random.seed(0)
    A = np.maximum(np.random.randn(5, 2), 0)
    b = np.maximum(np.random.randn(5), 0)
    x = cp.Variable(2)
    p = cp.Problem(cp.Maximize(-cp.sum(x)), [x >= 0, A @ x <= b])
    _solve(p, solver, use_quad_obj)
    np.testing.assert_allclose(x.value, [0., 0.], atol=1e-3)
    check_kkt(p, places=3)


@pytest.mark.parametrize("solver,use_quad_obj", QP_PARAMS)
def test_quad_form_bound(solver, use_quad_obj):
    P = np.array([[13, 12, -2], [12, 17, 6], [-2, 6, 12]])
    q = np.array([[-22], [-14.5], [13]])
    y_star = np.array([1., 0.5, -1.])
    y = cp.Variable(3)
    p = cp.Problem(cp.Minimize(0.5 * cp.QuadForm(y, P) + q.T @ y + 1),
                [y >= -1, y <= 1])
    _solve(p, solver, use_quad_obj)
    np.testing.assert_allclose(y.value, y_star, atol=1e-4)
    check_kkt(p)


@pytest.mark.parametrize("solver,use_quad_obj", QP_PARAMS)
def test_regression_1(solver, use_quad_obj):
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
    slope = cp.Variable(1)
    offset = cp.Variable(1)
    line = offset + x_data * slope
    p = cp.Problem(cp.Minimize(cp.sum_squares(line.T - y_data)))
    _solve(p, solver, use_quad_obj)
    np.testing.assert_allclose(1171.60037715, p.value, atol=1e-4)


@pytest.mark.parametrize("solver,use_quad_obj", QP_PARAMS)
def test_regression_2(solver, use_quad_obj):
    np.random.seed(1)
    n = 100
    true_coeffs = np.array([2, -2, 0.5])
    x_data = np.random.rand(n) * 5
    x_data_expanded = np.vstack([np.power(x_data, i) for i in range(1, 4)])
    y_data = x_data_expanded.T.dot(true_coeffs) + 0.5 * np.random.rand(n)
    slope = cp.Variable(1)
    offset = cp.Variable(1)
    quadratic_coeff = cp.Variable(1)
    quadratic = offset + x_data * slope + quadratic_coeff * np.power(x_data, 2)
    p = cp.Problem(cp.Minimize(cp.sum_squares(quadratic.T - y_data)))
    _solve(p, solver, use_quad_obj)
    np.testing.assert_allclose(139.225660756, p.value, atol=1e-4)


@pytest.mark.parametrize("solver,use_quad_obj", QP_PARAMS)
def test_control(solver, use_quad_obj):
    T = 30
    h = 0.1
    mass = 1
    drag = 0.1
    g = np.array([0, -9.8])
    position = cp.Variable((2, T))
    velocity = cp.Variable((2, T))
    force = cp.Variable((2, T - 1))
    constraints = []
    for i in range(T - 1):
        constraints += [position[:, i + 1] == position[:, i] + h * velocity[:, i]]
        acceleration = force[:, i] / mass + g - drag * velocity[:, i]
        constraints += [velocity[:, i + 1] == velocity[:, i] + h * acceleration]
    constraints += [position[:, 0] == 0]
    constraints += [position[:, -1] == np.array([100, 100])]
    constraints += [velocity[:, 0] == np.array([-20, 100])]
    constraints += [velocity[:, -1] == 0]
    p = cp.Problem(cp.Minimize(.01 * cp.sum_squares(force)), constraints)
    _solve(p, solver, use_quad_obj)
    np.testing.assert_allclose(1059.616, p.value, atol=1e-1)
    # KKT check skipped: check_stationary_lagrangian fails for 2D matrix
    # variables due to inconsistent gradient ordering (sum_squares uses
    # C order, constraint terms use F order). TODO fix this


@pytest.mark.parametrize("solver,use_quad_obj", QP_PARAMS)
def test_sparse_system(solver, use_quad_obj):
    m, n = 100, 80
    np.random.seed(1)
    A = sp.random_array((m, n), density=0.4)
    b = np.random.randn(m)
    xs = cp.Variable(n)
    p = cp.Problem(cp.Minimize(cp.sum_squares(A @ xs - b)), [xs == 0])
    _solve(p, solver, use_quad_obj)
    np.testing.assert_allclose(b.T.dot(b), p.value, atol=1e-4)


@pytest.mark.parametrize("solver,use_quad_obj", QP_PARAMS)
def test_smooth_ridge(solver, use_quad_obj):
    np.random.seed(1)
    n = 50
    k = 20
    A = np.ones((k, n))
    b = np.ones(k)
    xsr = cp.Variable(n)
    obj = cp.sum_squares(A @ xsr - b) + cp.sum_squares(xsr[:-1] - xsr[1:])
    p = cp.Problem(cp.Minimize(obj), [])
    _solve(p, solver, use_quad_obj)
    np.testing.assert_allclose(0, p.value, atol=1e-4)


@pytest.mark.parametrize("solver,use_quad_obj", QP_PARAMS)
def test_huber_small(solver, use_quad_obj):
    # The conic + use_quad_obj path has slightly looser precision.
    places = 3 if use_quad_obj else 4
    x = cp.Variable(3)
    objective = cp.sum(cp.huber(x))
    p = cp.Problem(cp.Minimize(objective), [x[2] >= 3])
    _solve(p, solver, use_quad_obj)
    atol = 10 ** (-places)
    np.testing.assert_allclose(3, x.value[2], atol=atol)
    np.testing.assert_allclose(5, objective.value, atol=atol)
    check_kkt(p, places=places)


@pytest.mark.parametrize("solver,use_quad_obj", QP_PARAMS)
def test_huber(solver, use_quad_obj):
    # Expected values below were computed with this seed; do not change it.
    np.random.seed(1)
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
    x = cp.Variable(n)
    objective = cp.sum(cp.huber(A @ x - b))
    p = cp.Problem(cp.Minimize(objective))
    _solve(p, solver, use_quad_obj)
    np.testing.assert_allclose(1.452797819667, objective.value, atol=1e-3)
    np.testing.assert_allclose(
        x.value, [1.20524645, -0.85271489, -0.50838494], atol=1e-3
    )


def _equivalent_forms_setup(seed=1):
    np.random.seed(seed)
    m, n, r = 100, 80, 70
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    G = np.random.randn(r, n)
    h = np.random.randn(r)
    return A, b, G, h


@pytest.mark.parametrize("solver,use_quad_obj", QP_PARAMS)
def test_equivalent_forms_1(solver, use_quad_obj):
    A, b, G, h = _equivalent_forms_setup()
    xef = cp.Variable(A.shape[1])
    p = cp.Problem(cp.Minimize(.1 * cp.sum((A @ xef - b) ** 2)), [G @ xef == h])
    _solve(p, solver, use_quad_obj)
    np.testing.assert_allclose(p.value, 68.1119420108, atol=1e-4)
    check_kkt(p, places=4)


@pytest.mark.parametrize("solver,use_quad_obj", QP_PARAMS)
def test_equivalent_forms_2(solver, use_quad_obj):
    A, b, G, h = _equivalent_forms_setup()
    # ||Ax-b||^2 = x^T (A^T A) x - 2(A^T b)^T x + ||b||^2
    P = A.T @ A
    q = -2 * A.T @ b
    r = b.T @ b
    xef = cp.Variable(A.shape[1])
    obj = .1 * (cp.QuadForm(xef, P) + q.T @ xef + r)
    p = cp.Problem(cp.Minimize(obj), [G @ xef == h])
    _solve(p, solver, use_quad_obj)
    np.testing.assert_allclose(p.value, 68.1119420108, atol=1e-4)
    check_kkt(p, places=4)


@pytest.mark.parametrize("solver,use_quad_obj", QP_PARAMS)
def test_equivalent_forms_3(solver, use_quad_obj):
    if solver == cp.KNITRO:
        pytest.skip("KNITRO does not support matrix_frac")
    A, b, G, h = _equivalent_forms_setup()
    P = A.T @ A
    q = -2 * A.T @ b
    r = b.T @ b
    Pinv = np.linalg.inv(P)
    xef = cp.Variable(A.shape[1])
    obj = .1 * (cp.matrix_frac(xef, Pinv) + q.T @ xef + r)
    p = cp.Problem(cp.Minimize(obj), [G @ xef == h])
    _solve(p, solver, use_quad_obj)
    np.testing.assert_allclose(p.value, 68.1119420108, atol=1e-4)
    check_kkt(p, places=4)


# --- Other native-QP-only correctness tests --- #

@pytest.mark.parametrize("solver", QP_SOLVER_PARAMS)
def test_qp_bound_attr(solver):
    solver_cls = SOLVER_MAP_QP.get(solver)
    if solver_cls is None or not getattr(solver_cls, "BOUNDED_VARIABLES", False):
        pytest.skip(f"{solver} does not support bounded-variable attributes")
    StandardTestQPs.test_qp_bound_attr(solver=solver)


@pytest.mark.parametrize("solver", QP_SOLVER_PARAMS)
def test_parametric(solver):
    """Solve parametric problem vs full problem."""
    x = cp.Variable()
    a = 10
    b_vec = [-10, -2.]

    x_full, obj_full = [], []
    for b in b_vec:
        prob = cp.Problem(cp.Minimize(a * (x ** 2) + b * x), [0 <= x, x <= 1])
        prob.solve(solver=solver)
        x_full.append(x.value)
        obj_full.append(prob.value)

    x_param, obj_param = [], []
    b = cp.Parameter()
    prob = cp.Problem(cp.Minimize(a * (x ** 2) + b * x), [0 <= x, x <= 1])
    for b_value in b_vec:
        b.value = b_value
        prob.solve(solver=solver)
        x_param.append(x.value)
        obj_param.append(prob.value)

    for i in range(len(b_vec)):
        np.testing.assert_allclose(x_param[i], x_full[i], atol=1e-3)
        np.testing.assert_allclose(obj_param[i], obj_full[i], atol=1e-5)


# --- Solver-specific tests --- #

@pytest.mark.parametrize(
    "solver",
    [s for s in [cp.OSQP, cp.QPALM, cp.HIGHS, cp.PIQP, cp.COPT]
     if s in INSTALLED_SOLVERS],
)
def test_warm_start(solver):
    """Re-solving with ``warm_start`` toggled on/off should return the same
    optimum. Covers OSQP, QPALM, HIGHS, PIQP, COPT -- XPRESS and GUROBI have
    their own warm-start tests below because they exercise different code
    paths (integer var / inspecting the gurobipy ``Model``).
    """
    m, n = 200, 100
    np.random.seed(1)
    A = np.random.randn(m, n)
    b = cp.Parameter(m)
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))

    b.value = np.random.randn(m)
    result = prob.solve(solver=solver, warm_start=False)
    result2 = prob.solve(solver=solver, warm_start=True)
    np.testing.assert_allclose(result, result2, atol=1e-5)
    b.value = np.random.randn(m)
    result = prob.solve(solver=solver, warm_start=True)
    result2 = prob.solve(solver=solver, warm_start=False)
    np.testing.assert_allclose(result, result2, atol=1e-5)


@pytest.mark.skipif(cp.GUROBI not in INSTALLED_SOLVERS, reason="GUROBI is not installed")
def test_gurobi_warmstart():
    """Test Gurobi warm start with a user provided point."""
    import gurobipy
    m, n = 4, 3
    y = cp.Variable(nonneg=True)
    X = cp.Variable((m, n))
    X_vals = np.reshape(np.arange(m * n), (m, n))
    prob = cp.Problem(cp.Minimize(y ** 2 + cp.sum(X)), [X == X_vals])
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
    b = cp.Parameter(m)
    x = cp.Variable(n, integer=True)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))

    b.value = np.random.randn(m)
    result = prob.solve(solver=cp.XPRESS, warm_start=False)
    result2 = prob.solve(solver=cp.XPRESS, warm_start=True)
    np.testing.assert_allclose(result, result2, atol=1e-5)
    x.value = x.value.astype(np.int64)

    xprime = cp.Variable(n, integer=True)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ xprime - b)))
    xprime.value = x.value
    result = prob.solve(solver=cp.XPRESS, warm_start=True)
    result2 = prob.solve(solver=cp.XPRESS, warm_start=False)
    np.testing.assert_allclose(result, result2, atol=1e-5)


@pytest.mark.skipif(not is_xpress_available(), reason="XPRESS license not available")
def test_xpress_duplicate_variable_names_qp():
    """Two variables sharing a name() must not crash the QP interface.

    cvxpy derives Xpress column names from ``Variable.name()``, so two variables
    created with the same name previously produced duplicate columns, which
    Xpress >= 9.5 rejects with ``?1030 Duplicate column names are not allowed``.
    The quadratic objective routes the problem through ``xpress_qpif``.
    """
    x = cp.Variable(3, name="dup")
    y = cp.Variable(3, name="dup")
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x - 1) + cp.sum_squares(y - 2)))
    prob.solve(solver=cp.XPRESS)
    assert prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    np.testing.assert_allclose(x.value, 1, atol=1e-4)
    np.testing.assert_allclose(y.value, 2, atol=1e-4)


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


def test_square_param():
    """Issue arising with square plus parameter."""
    a = cp.Parameter(value=1)
    b = cp.Variable()
    prob = cp.Problem(cp.Minimize(b ** 2 + cp.abs(a)))
    prob.solve(solver="SCS")
    np.testing.assert_allclose(prob.value, 1.0, atol=1e-5)


def test_gurobi_time_limit_no_solution():
    """If Gurobi hits its time limit before finding a solution, the solve
    must return cleanly and expose solver stats.
    """
    if cp.GUROBI in INSTALLED_SOLVERS:
        import gurobipy
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(x[0]), [x[0] >= 1])
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
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.norm(x, 1)), [x == 0])
        with pytest.raises(Exception) as exc_info:
            prob.solve(solver=cp.GUROBI, TimeLimit=0)
        assert str(exc_info.value) == f"The solver {cp.GUROBI} is not installed."


def test_gurobi_environment():
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
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.norm(x, 1)), [x == 0])
        with pytest.raises(Exception) as exc_info:
            prob.solve(solver=cp.GUROBI, TimeLimit=0)
        assert str(exc_info.value) == f"The solver {cp.GUROBI} is not installed."


_INFEAS_QP_SOLVERS = [
    s for s in [cp.OSQP, cp.HIGHS] if s in INSTALLED_SOLVERS
]


@pytest.mark.parametrize("solver", _INFEAS_QP_SOLVERS)
def test_infeasible_lp_ineq_constraints(solver):
    StandardTestInfeasibleProblems.test_lp_ineq_constraints(solver=solver)


@pytest.mark.parametrize("solver", _INFEAS_QP_SOLVERS)
def test_infeasible_lp_eq_constraints(solver):
    StandardTestInfeasibleProblems.test_lp_eq_constraints(solver=solver)


@pytest.mark.skipif(cp.HIGHS not in INSTALLED_SOLVERS, reason="HIGHS is not installed")
def test_highs_dense_quad_form():
    """Regression test for https://github.com/cvxpy/cvxpy/issues/3301
    and a related silent wrong-answer bug on highspy < 1.14.0.

    A dense quad_form applied to a linear expression (not a raw Variable)
    produces a Hessian whose upper and lower triangles may differ by
    floating-point epsilon after canonicalization. We pass it to HiGHS
    in triangular format. HiGHS < 1.14.0 only honors the lower triangle
    in that format and silently returns wrong solutions when given the
    upper triangle, hence the ``highspy >= 1.14.0`` minimum in
    pyproject.toml. Cross-check the objective against CLARABEL because
    status alone (``kOptimal``) does not detect a wrong-QP solve.
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
    highs_value = prob.value

    prob.solve(solver=cp.CLARABEL)
    assert prob.status == cp.OPTIMAL
    np.testing.assert_allclose(highs_value, prob.value, atol=1e-4)


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
    np.testing.assert_allclose(
        prob.solve(solver='MPAX', warm_start=False), -9, atol=1e-4
    )
    np.testing.assert_allclose(
        prob.solve(solver='MPAX', warm_start=True), -9, atol=1e-4
    )


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
