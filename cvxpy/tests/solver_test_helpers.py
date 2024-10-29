"""
Copyright 2019, the CVXPY developers.

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
import warnings

import numpy as np

import cvxpy as cp
from cvxpy.tests.base_test import BaseTest


class SolverTestHelper:

    def __init__(self, obj_pair, var_pairs, con_pairs) -> None:
        self.objective = obj_pair[0]
        self.constraints = [c for c, _ in con_pairs]
        self.prob = cp.Problem(self.objective, self.constraints)
        self.variables = [x for x, _ in var_pairs]

        self.expect_val = obj_pair[1]
        self.expect_dual_vars = [dv for _, dv in con_pairs]
        self.expect_prim_vars = [pv for _, pv in var_pairs]
        self.tester = BaseTest()

    def solve(self, solver, **kwargs) -> None:
        self.prob.solve(solver=solver, **kwargs)

    def check_primal_feasibility(self, places) -> None:
        all_cons = [c for c in self.constraints]  # shallow copy
        for x in self.prob.variables():
            attrs = x.attributes
            if attrs['nonneg'] or attrs['pos']:
                all_cons.append(x >= 0)
            elif attrs['nonpos'] or attrs['neg']:
                all_cons.append(x <= 0)
            elif attrs['imag']:
                all_cons.append(x + cp.conj(x) == 0)
            elif attrs['symmetric']:
                all_cons.append(x == x.T)
            elif attrs['diag']:
                all_cons.append(x - cp.diag(cp.diag(x)) == 0)
            elif attrs['PSD']:
                all_cons.append(x >> 0)
            elif attrs['NSD']:
                all_cons.append(x << 0)
            elif attrs['hermitian']:
                all_cons.append(x == cp.conj(x.T))
            elif attrs['boolean'] or attrs['integer']:
                round_val = np.round(x.value)
                all_cons.append(x == round_val)
        for con in all_cons:
            viol = con.violation()
            if isinstance(viol, np.ndarray):
                viol = np.linalg.norm(viol, ord=2)
            self.tester.assertAlmostEqual(viol, 0, places)

    def check_dual_domains(self, places) -> None:
        # A full "dual feasibility" check would involve checking a stationary Lagrangian.
        # No such test is planned here.
        #
        # TODO: once dual variables are stored for attributes
        #   (e.g. X = Variable(shape=(n,n), PSD=True)), check
        #   domains for dual variables of the attribute constraint.
        for con in self.constraints:
            if isinstance(con, cp.constraints.Cone):
                dual_violation = con.dual_residual
                if isinstance(con, cp.constraints.SOC):
                    dual_violation = np.linalg.norm(dual_violation)
                self.tester.assertLessEqual(dual_violation, 10**(-places))
            elif isinstance(con, cp.constraints.Inequality):
                # TODO: move this to Inequality.dual_violation
                dv = con.dual_value
                min_dv = np.min(dv)
                self.tester.assertGreaterEqual(min_dv, -(10**(-places)))
            elif isinstance(con, (cp.constraints.Equality, cp.constraints.Zero)):
                dv = con.dual_value
                self.tester.assertIsNotNone(dv)
                if isinstance(dv, np.ndarray):
                    contents = dv.dtype
                    self.tester.assertEqual(contents, float)
                else:
                    self.tester.assertIsInstance(dv, float)
            else:
                raise ValueError('Unknown constraint type %s.' % type(con))

    def check_complementarity(self, places) -> None:
        # TODO: once dual variables are stored for attributes
        #   (e.g. X = Variable(shape=(n,n), PSD=True)), check
        #   complementarity against the dual variable of the
        #   attribute constraint.
        for con in self.constraints:
            if isinstance(con, (cp.constraints.Inequality,
                                cp.constraints.Equality)):
                comp = cp.vdot(con.expr, con.dual_value).value
            elif isinstance(con, (cp.constraints.ExpCone,
                                  cp.constraints.SOC,
                                  cp.constraints.NonNeg,
                                  cp.constraints.Zero,
                                  cp.constraints.PSD,
                                  cp.constraints.PowCone3D,
                                  cp.constraints.PowConeND)):
                comp = cp.vdot(con.args, con.dual_value).value
            elif isinstance(con, cp.RelEntrConeQuad) or isinstance(con, cp.OpRelEntrConeQuad):
                msg = '\nDual variables not implemented for quadrature based approximations;' \
                       + '\nSkipping complementarity check.'
                warnings.warn(msg)
            else:
                raise ValueError('Unknown constraint type %s.' % type(con))
            self.tester.assertAlmostEqual(comp, 0, places)
            
    def check_stationary_lagrangian(self, places) -> None:
        L = self.prob.objective.expr
        objective = self.prob.objective
        if objective.NAME == 'minimize':
            L = objective.expr
        else:
            L = -objective.expr
        for con in self.constraints:
            if isinstance(con, (cp.constraints.Inequality,
                                cp.constraints.Equality)):
                dual_var_value = con.dual_value
                prim_var_expr = con.expr
                L = L + cp.vdot(dual_var_value, prim_var_expr)
            elif isinstance(con, (cp.constraints.ExpCone,
                                  cp.constraints.SOC,
                                  cp.constraints.Zero,
                                  cp.constraints.NonNeg,
                                  cp.constraints.PSD,
                                  cp.constraints.PowCone3D,
                                  cp.constraints.PowConeND)):
                L = L - cp.vdot(con.args, con.dual_value)
            else:
                raise NotImplementedError()
        try:
            g = L.grad
        except TypeError as e:
            assert 'is not subscriptable' in str(e)
            msg = """\n
            CVXPY problems with `diag` variables are not supported for
            stationarity checks as of now
            """
            self.tester.fail(msg)
        bad_norms = []

        r"""The convention that we follow for construting the Lagrangian is: 1) Move all
        explicitly passed constraints to the problem (via Problem.constraints) into the
        Lagrangian --- dLdX == 0 for any such variables 2) Constraints that have
        implicitly been imposed on variables at the time of declaration via specific
        flags (e.g.: PSD/symmetric etc.), in such a case we check, `dLdX\in K^{*}`, where
        `K` is the convex cone corresponding to the implicit constraint on `X`
        """
        for (opt_var, v) in g.items():
            if all(not attr for attr in list(map(lambda x: x[1], opt_var.attributes.items()))):
                """Case when the variable doesn't have any special attributes"""
                norm = np.linalg.norm(v.data) / np.sqrt(opt_var.size)
                if norm > 10**(-places):
                    bad_norms.append((norm, opt_var))
            else:
                if opt_var.is_psd():
                    """The PSD cone is self-dual"""
                    g_bad_mat = cp.Constant(np.reshape(g[opt_var].toarray(), opt_var.shape))
                    tmp_con = g_bad_mat >> 0
                    dual_cone_violation = tmp_con.residual
                    if dual_cone_violation > 10**(-places):
                        bad_norms.append((dual_cone_violation, opt_var))
                elif opt_var.is_nsd():
                    """The NSD cone is also self-dual"""
                    g_bad_mat = cp.Constant(np.reshape(g[opt_var].toarray(), opt_var.shape))
                    tmp_con = g_bad_mat << 0
                    dual_cone_violation = tmp_con.residual
                    if dual_cone_violation > 10**(-places):
                        bad_norms.append((dual_cone_violation, opt_var))
                elif opt_var.is_diag():
                    """The dual cone to the set of diagonal matrices is the set of
                        'Hollow' matrices i.e. matrices with diagonal entries zero"""
                    g_bad_mat = np.reshape(g[opt_var].toarray(), opt_var.shape)
                    diag_entries = np.diag(opt_var.value)
                    dual_cone_violation = np.linalg.norm(diag_entries) / np.sqrt(opt_var.size)
                    if diag_entries > 10**(-places):
                        bad_norms.append((dual_cone_violation, opt_var))
                elif opt_var.is_symmetric():
                    r"""The dual cone to the set of symmetric matrices is the
                    set of skew-symmetric matrices, so we check if dLdX \in
                    set(skew-symmetric-matrices)
                    g[opt_var] is the problematic gradient in question"""
                    g_bad_mat = np.reshape(g[opt_var].toarray(), opt_var.shape)
                    mat = g_bad_mat + g_bad_mat.T
                    dual_cone_violation = np.linalg.norm(mat) / np.sqrt(opt_var.size)
                    if dual_cone_violation > 10**(-places):
                        bad_norms.append((dual_cone_violation, opt_var))
                elif opt_var.is_nonpos():
                    """The cone of matrices with all entries nonpos is self-dual"""
                    g_bad_mat = cp.Constant(np.reshape(g[opt_var].toarray(), opt_var.shape))
                    tmp_con = g_bad_mat <= 0
                    dual_cone_violation = np.linalg.norm(tmp_con.residual) / np.sqrt(opt_var.size)
                    if dual_cone_violation > 10**(-places):
                        bad_norms.append((dual_cone_violation, opt_var))
                elif opt_var.is_nonneg():
                    """The cone of matrices with all entries nonneg is self-dual"""
                    g_bad_mat = cp.Constant(np.reshape(g[opt_var].toarray(), opt_var.shape))
                    tmp_con = g_bad_mat >= 0
                    dual_cone_violation = np.linalg.norm(tmp_con.residual) / np.sqrt(opt_var.size)
                    if dual_cone_violation > 10**(-places):
                        bad_norms.append((dual_cone_violation, opt_var))

        if len(bad_norms):
            msg = f"""\n
        The gradient of Lagrangian with respect to the primal variables
        is above the threshold of 10^{-places}. The names of the problematic
        variables and the corresponding gradient norms are as follows:
            """
            for norm, opt_var in bad_norms:
                msg += f"\n\t\t\t{opt_var.name} : {norm}"
            msg += '\n'
            self.tester.fail(msg)
        pass 

    def verify_objective(self, places) -> None:
        actual = self.prob.value
        expect = self.expect_val
        if expect is not None:
            self.tester.assertAlmostEqual(actual, expect, places)

    def verify_primal_values(self, places) -> None:
        for idx in range(len(self.variables)):
            actual = self.variables[idx].value
            expect = self.expect_prim_vars[idx]
            if expect is not None:
                self.tester.assertItemsAlmostEqual(actual, expect, places)

    def verify_dual_values(self, places) -> None:
        for idx in range(len(self.constraints)):
            actual = self.constraints[idx].dual_value
            expect = self.expect_dual_vars[idx]
            if expect is not None:
                if isinstance(actual, list):
                    for i in range(len(actual)):
                        act = actual[i]
                        exp = expect[i]
                        self.tester.assertItemsAlmostEqual(act, exp, places)
                else:
                    self.tester.assertItemsAlmostEqual(actual, expect, places)


def lp_0() -> SolverTestHelper:
    x = cp.Variable(shape=(2,))
    con_pairs = [(x == 0, None)]
    obj_pair = (cp.Minimize(cp.norm(x, 1) + 1.0), 1)
    var_pairs = [(x, np.array([0, 0]))]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def lp_1() -> SolverTestHelper:
    # Example from
    # http://cvxopt.org/userguide/coneprog.html?highlight=solvers.lp#cvxopt.solvers.lp
    x = cp.Variable(shape=(2,), name='x')
    objective = cp.Minimize(-4 * x[0] - 5 * x[1])
    constraints = [2 * x[0] + x[1] <= 3,
                   x[0] + 2 * x[1] <= 3,
                   x[0] >= 0,
                   x[1] >= 0]
    con_pairs = [(constraints[0], 1),
                 (constraints[1], 2),
                 (constraints[2], 0),
                 (constraints[3], 0)]
    var_pairs = [(x, np.array([1, 1]))]
    obj_pair = (objective, -9)
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def lp_2() -> SolverTestHelper:
    x = cp.Variable(shape=(2,), name='x')
    objective = cp.Minimize(x[0] + 0.5 * x[1])
    constraints = [x[0] >= -100, x[0] <= -10, x[1] == 1]
    con_pairs = [(constraints[0], 1),
                 (constraints[1], 0),
                 (constraints[2], -0.5)]
    var_pairs = [(x, np.array([-100, 1]))]
    obj_pair = (objective, -99.5)
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def lp_3() -> SolverTestHelper:
    # an unbounded problem
    x = cp.Variable(5)
    objective = (cp.Minimize(cp.sum(x)), -np.inf)
    var_pairs = [(x, None)]
    con_pairs = [(x <= 1, None)]
    sth = SolverTestHelper(objective, var_pairs, con_pairs)
    return sth


def lp_4() -> SolverTestHelper:
    # an infeasible problem
    x = cp.Variable(5)
    objective = (cp.Minimize(cp.sum(x)), np.inf)
    var_pairs = [(x, None)]
    con_pairs = [(x <= 0, None),
                 (x >= 1, None)]
    sth = SolverTestHelper(objective, var_pairs, con_pairs)
    return sth


def lp_5() -> SolverTestHelper:
    # a problem with redundant equality constraints.
    #
    # 10 variables, 6 equality constraints A @ x == b (two redundant)
    x0 = np.array([0, 1, 0, 2, 0, 4, 0, 5, 6, 7])
    mu0 = np.array([-2, -1, 0, 1, 2, 3.5])
    np.random.seed(0)
    A_min = np.random.randn(4, 10)
    A_red = A_min.T @ np.random.rand(4, 2)
    A_red = A_red.T
    A = np.vstack((A_min, A_red))
    b = A @ x0  # x0 is primal feasible
    c = A.T @ mu0  # mu0 is dual-feasible
    c[[0, 2, 4, 6]] += np.random.rand(4)
    # ^  c >= A.T @ mu0 exhibits complementary slackness with respect to x0
    #    Therefore (x0, mu0) are primal-dual optimal for ...
    x = cp.Variable(10)
    objective = (cp.Minimize(c @ x), c @ x0)
    var_pairs = [(x, x0)]
    con_pairs = [(x >= 0, None), (A @ x == b, None)]
    sth = SolverTestHelper(objective, var_pairs, con_pairs)
    return sth


def lp_6() -> SolverTestHelper:
    """Test LP with no constraints"""
    x = cp.Variable()
    from cvxpy.expressions.constants import Constant
    objective = cp.Maximize(Constant(0.23) * x)
    obj_pair = (objective, np.inf)
    var_pairs = [(x, None)]
    sth = SolverTestHelper(obj_pair, var_pairs, [])
    return sth


def lp_7() -> SolverTestHelper:
    """
    An ill-posed problem to test multiprecision ability of solvers.

    This test will not pass on CVXOPT (as of v1.3.1) and on SDPA without GMP support.
    """
    n = 50
    a = cp.Variable((n+1))
    delta = cp.Variable((n))
    b = cp.Variable((n+1))
    objective = cp.Minimize(cp.sum(cp.pos(delta)))
    constraints = [
        a[1:] - a[:-1] == delta,
        a >= cp.pos(b),
    ]
    con_pairs = [(constraints[0], None),
                 (constraints[1], None)]
    var_pairs = [(a, None),
                 (delta, None),
                 (b, None)]
    obj_pair = (objective, 0.)
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def qp_0() -> SolverTestHelper:
    # univariate feasible problem
    x = cp.Variable(1)
    objective = cp.Minimize(cp.square(x))
    constraints = [x[0] >= 1]
    con_pairs = [(constraints[0], 2)]
    obj_pair = (objective, 1)
    var_pairs = [(x, 1)]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def socp_0() -> SolverTestHelper:
    x = cp.Variable(shape=(2,))
    obj_pair = (cp.Minimize(cp.norm(x, 2) + 1), 1)
    con_pairs = [(x == 0, None)]
    var_pairs = [(x, np.array([0, 0]))]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def socp_1() -> SolverTestHelper:
    """
    min 3 * x[0] + 2 * x[1] + x[2]
    s.t. norm(x,2) <= y
         x[0] + x[1] + 3*x[2] >= 1.0
         y <= 5
    """
    x = cp.Variable(shape=(3,))
    y = cp.Variable()
    soc = cp.constraints.second_order.SOC(y, x)
    constraints = [soc,
                   x[0] + x[1] + 3 * x[2] >= 1.0,
                   y <= 5]
    obj = cp.Minimize(3 * x[0] + 2 * x[1] + x[2])
    expect_x = np.array([-3.874621860638774, -2.129788233677883, 2.33480343377204])
    expect_x = np.round(expect_x, decimals=5)
    expect_y = 5
    var_pairs = [(x, expect_x),
                 (y, expect_y)]
    expect_soc = [np.array([2.86560262]), np.array([2.22062583,  1.22062583, -1.33812252])]
    expect_ineq1 = 0.7793969212001993
    expect_ineq2 = 2.865602615049077
    con_pairs = [(constraints[0], expect_soc),
                 (constraints[1], expect_ineq1),
                 (constraints[2], expect_ineq2)]
    obj_pair = (obj, -13.548638904065102)
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def socp_2() -> SolverTestHelper:
    """
    An (unnecessarily) SOCP-based reformulation of LP_1.
    """
    x = cp.Variable(shape=(2,), name='x')
    objective = cp.Minimize(-4 * x[0] - 5 * x[1])
    expr = cp.reshape(x[0] + 2 * x[1], (1, 1), order='F')
    constraints = [2 * x[0] + x[1] <= 3,
                   cp.constraints.SOC(cp.Constant([3]), expr),
                   x[0] >= 0,
                   x[1] >= 0]
    con_pairs = [(constraints[0], 1),
                 (constraints[1], [np.array([2.]), np.array([[-2.]])]),
                 (constraints[2], 0),
                 (constraints[3], 0)]
    var_pairs = [(x, np.array([1, 1]))]
    obj_pair = (objective, -9)
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def socp_3(axis) -> SolverTestHelper:
    x = cp.Variable(shape=(2,))
    c = np.array([-1, 2])
    root2 = np.sqrt(2)
    u = np.array([[1 / root2, -1 / root2], [1 / root2, 1 / root2]])
    mat1 = np.diag([root2, 1 / root2]) @ u.T
    mat2 = np.diag([1, 1])
    mat3 = np.diag([0.2, 1.8])

    X = cp.vstack([mat1 @ x, mat2 @ x, mat3 @ x])  # stack these as rows
    t = cp.Constant(np.ones(3, ))
    objective = cp.Minimize(c @ x)
    if axis == 0:
        con = cp.constraints.SOC(t, X.T, axis=0)
        con_expect = [
            np.array([0, 1.16454469e+00, 7.67560451e-01]),
            np.array([[0, -9.74311819e-01, -1.28440860e-01],
                      [0, 6.37872081e-01, 7.56737724e-01]])
        ]
    else:
        con = cp.constraints.SOC(t, X, axis=1)
        con_expect = [
            np.array([0, 1.16454469e+00, 7.67560451e-01]),
            np.array([[0, 0],
                      [-9.74311819e-01, 6.37872081e-01],
                      [-1.28440860e-01, 7.56737724e-01]])
        ]
    obj_pair = (objective, -1.932105)
    con_pairs = [(con, con_expect)]
    var_pairs = [(x, np.array([0.83666003, -0.54772256]))]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def sdp_1(objective_sense) -> SolverTestHelper:
    """
    Solve "Example 8.3" from Convex Optimization by Boyd & Vandenberghe.

    Verify (1) optimal objective values, (2) that the dual variable to the PSD constraint
    belongs to the correct cone (i.e. the dual variable is itself PSD), and (3) that
    complementary slackness holds with the PSD primal variable and its dual variable.
    """
    rho = cp.Variable(shape=(4, 4), symmetric=True)
    constraints = [0.6 <= rho[0, 1], rho[0, 1] <= 0.9,
                   0.8 <= rho[0, 2], rho[0, 2] <= 0.9,
                   0.5 <= rho[1, 3], rho[1, 3] <= 0.7,
                   -0.8 <= rho[2, 3], rho[2, 3] <= -0.4,
                   rho[0, 0] == 1, rho[1, 1] == 1, rho[2, 2] == 1, rho[3, 3] == 1,
                   rho >> 0]
    if objective_sense == 'min':
        obj = cp.Minimize(rho[0, 3])
        obj_pair = (obj, -0.39)
    elif objective_sense == 'max':
        obj = cp.Maximize(rho[0, 3])
        obj_pair = (obj, 0.23)
    else:
        raise RuntimeError('Unknown objective_sense.')
    con_pairs = [(c, None) for c in constraints]
    var_pairs = [(rho, None)]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def sdp_2() -> SolverTestHelper:
    """
    Example SDO2 from MOSEK 9.2 documentation.
    """
    X1 = cp.Variable(shape=(2, 2), symmetric=True)
    X2 = cp.Variable(shape=(4, 4), symmetric=True)
    C1 = np.array([[1, 0], [0, 6]])
    A1 = np.array([[1, 1], [1, 2]])
    C2 = np.array([[1, -3, 0, 0], [-3, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
    A2 = np.array([[0, 1, 0, 0], [1, -1, 0, 0], [0, 0, 0, 0], [0, 0, 0, -3]])
    b = 23
    k = -3
    var_pairs = [
        (X1, np.array([[21.04711571, 4.07709873],
                       [4.07709873, 0.7897868]])),
        (X2, np.array([[5.05366214, -3., 0., 0.],
                       [-3., 1.78088676, 0., 0.],
                       [0., 0., 0., 0.],
                       [0., 0., 0., -0.]]))
    ]
    con_pairs = [
        (cp.trace(A1 @ X1) + cp.trace(A2 @ X2) == b, -0.83772234),
        (X2[0, 1] <= k, 11.04455278),
        (X1 >> 0, np.array([[21.04711571, 4.07709873],
                            [4.07709873, 0.7897868]])),
        (X2 >> 0, np.array([[1., 1.68455405, 0., 0.],
                            [1.68455405, 2.83772234, 0., 0.],
                            [0., 0., 1., 0.],
                            [0., 0., 0., 2.51316702]]))
    ]
    obj_expr = cp.Minimize(cp.trace(C1 @ X1) + cp.trace(C2 @ X2))
    obj_pair = (obj_expr, 52.40127214)
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def expcone_1() -> SolverTestHelper:
    """
    min   3 * x[0] + 2 * x[1] + x[2]
    s.t.  0.1 <= x[0] + x[1] + x[2] <= 1
          x >= 0
          x[0] >= x[1] * exp(x[2] / x[1])
    """
    x = cp.Variable(shape=(3, 1))
    cone_con = cp.constraints.ExpCone(x[2], x[1], x[0])
    constraints = [cp.sum(x) <= 1.0,
                   cp.sum(x) >= 0.1,
                   x >= 0,
                   cone_con]
    obj = cp.Minimize(3 * x[0] + 2 * x[1] + x[2])
    obj_pair = (obj, 0.23534820622420757)
    expect_exp = [np.array([-1.35348213]), np.array([-0.35348211]), np.array([0.64651792])]
    con_pairs = [(constraints[0], 0),
                 (constraints[1], 2.3534821130067614),
                 (constraints[2], np.zeros(shape=(3, 1))),
                 (constraints[3], expect_exp)]
    expect_x = np.array([[0.05462721], [0.02609378], [0.01927901]])
    var_pairs = [(x, expect_x)]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def expcone_socp_1() -> SolverTestHelper:
    """
    A random risk-parity portfolio optimization problem.
    """
    sigma = np.array([[1.83, 1.79, 3.22],
                      [1.79, 2.18, 3.18],
                      [3.22, 3.18, 8.69]])
    L = np.linalg.cholesky(sigma)
    c = 0.75
    t = cp.Variable(name='t')
    x = cp.Variable(shape=(3,), name='x')
    s = cp.Variable(shape=(3,), name='s')
    e = cp.Constant(np.ones(3, ))
    objective = cp.Minimize(t - c * e @ s)
    con1 = cp.norm(L.T @ x, p=2) <= t
    con2 = cp.constraints.ExpCone(s, e, x)
    # SolverTestHelper data
    obj_pair = (objective, 4.0751197)
    var_pairs = [
        (x, np.array([0.576079, 0.54315, 0.28037])),
        (s, np.array([-0.55150, -0.61036, -1.27161])),
    ]
    con_pairs = [
        (con1, 1.0),
        (con2, [np.array([-0.75, -0.75, -0.75]),
                np.array([-1.16363, -1.20777, -1.70371]),
                np.array([1.30190, 1.38082, 2.67496])]
         )
    ]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def sdp_pcp_1() -> SolverTestHelper:
    """
    Example sdp and power cone.
    """
    Sigma = np.array([[ 0.4787481 , -0.96924914],
                      [-0.96924914,  2.77788598]])

    x = cp.Variable(shape=(2,1))
    y = cp.Variable(shape=(2,1))
    X = cp.Variable(shape=(2,2), symmetric=True)
    M1 = cp.vstack([X, x.T])
    M2 = cp.vstack([x, np.ones((1, 1))])
    M3 = cp.hstack([M1, M2])

    var_pairs = [
        (x, np.array([[0.72128204],
                      [0.27871796]])),
        (y, np.array([[0.01],
                      [0.01]])),
        (X, np.array([[0.52024779, 0.20103426],
                      [0.20103426, 0.0776837 ]])),        
    ]
    con_pairs = [
        (cp.sum(x) == 1, -0.1503204799112807),
        (x >= 0, np.array([[-0.],
                           [-0.]])),
        (y >= 0.01, np.array([[0.70705506],
                               [0.70715844]])),
        (M3 >> 0, np.array([[ 0.4787481 , -0.96924914, -0.07516024],
                            [-0.96924914,  2.77788598, -0.07516024],
                            [-0.07516024, -0.07516024,  0.07516094]])),
        (cp.PowCone3D(x, np.ones((2,1)), y, 0.9), [np.array([[1.17878172e-09],
                                                             [3.05162243e-09]]),
                                                   np.array([[9.20157640e-10],
                                                             [9.40823207e-10]]),
                                                   np.array([[2.41053358e-10],
                                                             [7.43432462e-10]])]),
        ]
    obj_expr = cp.Minimize(cp.trace(Sigma @ X) + cp.norm(y, p=2))
    obj_pair = (obj_expr, 0.089301671322676)
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def pcp_1() -> SolverTestHelper:
    """
    Use a 3D power cone formulation for

    min 3 * x[0] + 2 * x[1] + x[2]
    s.t. norm(x,2) <= y
         x[0] + x[1] + 3*x[2] >= 1.0
         y <= 5
    """
    x = cp.Variable(shape=(3,))
    y_square = cp.Variable()
    epis = cp.Variable(shape=(3,))
    constraints = [cp.constraints.PowCone3D(np.ones(3), epis, x, cp.Constant([0.5, 0.5, 0.5])),
                   cp.sum(epis) <= y_square,
                   x[0] + x[1] + 3 * x[2] >= 1.0,
                   y_square <= 25]
    obj = cp.Minimize(3 * x[0] + 2 * x[1] + x[2])
    expect_x = np.array([-3.874621860638774, -2.129788233677883, 2.33480343377204])
    expect_epis = expect_x ** 2
    expect_x = np.round(expect_x, decimals=5)
    expect_epis = np.round(expect_epis, decimals=5)
    expect_y_square = 25
    var_pairs = [(x, expect_x),
                 (epis, expect_epis),
                 (y_square, expect_y_square)]
    expect_ineq1 = 0.7793969212001993
    expect_ineq2 = 2.865602615049077 / 10
    expect_pc = [np.array([4.30209047, 1.29985494, 1.56211543]),
                 np.array([0.28655796, 0.28655796, 0.28655796]),
                 np.array([2.22062898, 1.22062899, -1.33811302])]
    con_pairs = [(constraints[0], expect_pc),
                 (constraints[1], expect_ineq2),
                 (constraints[2], expect_ineq1),
                 (constraints[3], expect_ineq2)]
    obj_pair = (obj, -13.548638904065102)
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def pcp_2() -> SolverTestHelper:
    """
    Reformulate

        max  (x**0.2)*(y**0.8) + z**0.4 - x
        s.t. x + y + z/2 == 2
             x, y, z >= 0
    Into

        max  x3 + x4 - x0
        s.t. x0 + x1 + x2 / 2 == 2,
             (x0, x1, x3) in Pow3D(0.2)
             (x2, 1.0, x4) in Pow3D(0.4)
    """
    x = cp.Variable(shape=(3,))
    hypos = cp.Variable(shape=(2,))
    objective = cp.Minimize(-cp.sum(hypos) + x[0])
    arg1 = cp.hstack([x[0], x[2]])
    arg2 = cp.hstack(([x[1], 1.0]))
    pc_con = cp.constraints.PowCone3D(arg1, arg2, hypos, [0.2, 0.4])
    expect_pc_con = [np.array([1.48466366, 0.24233184]),
                     np.array([0.48466367, 0.83801333]),
                     np.array([-1., -1.])]
    con_pairs = [
        (x[0] + x[1] + 0.5 * x[2] == 2, 0.4846636697795672),
        (pc_con, expect_pc_con)
    ]
    obj_pair = (objective, -1.8073406786220672)
    var_pairs = [
        (x, np.array([0.06393515, 0.78320961, 2.30571048])),
        (hypos, None)
    ]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def pcp_3() -> SolverTestHelper:
    from scipy.optimize import Bounds, minimize
    w = cp.Variable((2, 1))
    D = np.array([
       [-1.0856306,   0.99734545],
       [0.2829785,  -1.50629471],
       [-0.57860025,  1.65143654],
       [-2.42667924, -0.42891263],
       [1.26593626, -0.8667404],
       [-0.67888615, -0.09470897],
       [1.49138963, -0.638902]])  # T-by-N
    """
    Minimize ||D @ w||_p s.t. 0 <= w, sum(w) == 1.
        Refer to https://docs.mosek.com/modeling-cookbook/powo.html#p-norm-cones
    """
    p = 1/0.4
    T = D.shape[0]
    t = cp.Variable()
    d = cp.Variable((T, 1))
    ones = np.ones((T, 1))

    powcone = cp.constraints.PowCone3D(d, t * ones, D @ w, 1/p)
    constraints = [cp.sum(w) == 1, w >= 0, powcone, cp.sum(d) == t]
    con_pairs = [
        (constraints[0], -1.51430),
        (constraints[1], np.array([0.0, 0.0])),
        (constraints[2], [
              np.array([[0.40000935],
                        [0.40000935],
                        [0.40000935],
                        [0.40000935],
                        [0.40000935],
                        [0.40000935],
                        [0.40000935]]),
              np.array([[2.84369172e-03],
                        [1.22657446e-01],
                        [1.12146997e-01],
                        [3.45802205e-01],
                        [2.76327461e-05],
                        [1.27539057e-02],
                        [3.75878155e-03]]),
              np.array([[-0.04031276],
                        [0.38577107],
                        [-0.36558292],
                        [0.71847219],
                        [0.00249992],
                        [0.09919715],
                        [-0.04765863]])]),
        (constraints[3], 0.40000935)
    ]

    def univar_obj(w0):
        return np.linalg.norm(D[:, 0] * w0 + D[:, 1] * (1 - w0), ord=p)
    univar_bounds = Bounds([0], [1])
    univar_res = minimize(univar_obj, np.array([0.4]), bounds=univar_bounds, tol=1e-16)
    w_opt = np.array([[univar_res.x], [1 - univar_res.x]])

    obj_pair = (cp.Minimize(t), univar_res.fun)
    var_pairs = [(d, np.array([
                       [7.17144981e-03],
                       [3.09557056e-01],
                       [2.83038570e-01],
                       [8.72785905e-01],
                       [6.92995408e-05],
                       [3.21904516e-02],
                       [9.48918352e-03]])),
                 (w, w_opt),
                 (t, np.array([univar_res.fun]))]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def mi_lp_0() -> SolverTestHelper:
    x = cp.Variable(shape=(2,))
    bool_var = cp.Variable(boolean=True)
    con_pairs = [(x == bool_var, None),
                 (bool_var == 0, None)]
    obj_pair = (cp.Minimize(cp.norm(x, 1) + 1.0), 1)
    var_pairs = [(x, np.array([0, 0])),
                 (bool_var, 0)]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def mi_lp_1() -> SolverTestHelper:
    x = cp.Variable(2, name='x')
    boolvar = cp.Variable(boolean=True)
    intvar = cp.Variable(integer=True)
    objective = cp.Minimize(-4 * x[0] - 5 * x[1])
    constraints = [2 * x[0] + x[1] <= intvar,
                   x[0] + 2 * x[1] <= 3 * boolvar,
                   x >= 0,
                   intvar == 3 * boolvar,
                   intvar == 3]
    obj_pair = (objective, -9)
    var_pairs = [(x, np.array([1, 1])),
                 (boolvar, 1),
                 (intvar,  3)]
    con_pairs = [(c, None) for c in constraints]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def mi_lp_2() -> SolverTestHelper:
    # Instance "knapPI_1_50_1000_1" from "http://www.diku.dk/~pisinger/genhard.c"
    n = 50
    c = 995
    z = 8373
    coeffs = [[1, 94, 485, 0], [2, 506, 326, 0], [3, 416, 248, 0],
              [4, 992, 421, 0], [5, 649, 322, 0], [6, 237, 795, 0],
              [7, 457, 43, 1], [8, 815, 845, 0], [9, 446, 955, 0],
              [10, 422, 252, 0], [11, 791, 9, 1], [12, 359, 901, 0],
              [13, 667, 122, 1], [14, 598, 94, 1], [15, 7, 738, 0],
              [16, 544, 574, 0], [17, 334, 715, 0], [18, 766, 882, 0],
              [19, 994, 367, 0], [20, 893, 984, 0], [21, 633, 299, 0],
              [22, 131, 433, 0], [23, 428, 682, 0], [24, 700, 72, 1],
              [25, 617, 874, 0], [26, 874, 138, 1], [27, 720, 856, 0],
              [28, 419, 145, 0], [29, 794, 995, 0], [30, 196, 529, 0],
              [31, 997, 199, 1], [32, 116, 277, 0], [33, 908, 97, 1],
              [34, 539, 719, 0], [35, 707, 242, 0], [36, 569, 107, 0],
              [37, 537, 122, 0], [38, 931, 70, 1], [39, 726, 98, 1],
              [40, 487, 600, 0], [41, 772, 645, 0], [42, 513, 267, 0],
              [43, 81, 972, 0], [44, 943, 895, 0], [45, 58, 213, 0],
              [46, 303, 748, 0], [47, 764, 487, 0], [48, 536, 923, 0],
              [49, 724, 29, 1], [50, 789, 674, 0]]  # index, p / w / x
    X = cp.Variable(n, boolean=True)
    objective = cp.Maximize(cp.sum(cp.multiply([i[1] for i in coeffs], X)))
    constraints = [cp.sum(cp.multiply([i[2] for i in coeffs], X)) <= c]
    obj_pair = (objective, z)
    con_pairs = [(constraints[0], None)]
    var_pairs = [(X, None)]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def mi_lp_3() -> SolverTestHelper:
    # infeasible (but relaxable) test case
    x = cp.Variable(4, boolean=True)
    from cvxpy.expressions.constants import Constant
    objective = cp.Maximize(Constant(1))
    constraints = [x[0] + x[1] + x[2] + x[3] <= 2,
                   x[0] + x[1] + x[2] + x[3] >= 2,
                   x[0] + x[1] <= 1,
                   x[0] + x[2] <= 1,
                   x[0] + x[3] <= 1,
                   x[2] + x[3] <= 1,
                   x[1] + x[3] <= 1,
                   x[1] + x[2] <= 1]
    obj_pair = (objective, -np.inf)
    con_pairs = [(c, None) for c in constraints]
    var_pairs = [(x, None)]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def mi_lp_4() -> SolverTestHelper:
    """Test MI without constraints"""
    x = cp.Variable(boolean=True)
    from cvxpy.expressions.constants import Constant
    objective = cp.Maximize(Constant(0.23) * x)
    obj_pair = (objective, 0.23)
    var_pairs = [(x, 1)]
    sth = SolverTestHelper(obj_pair, var_pairs, [])
    return sth


def mi_lp_5() -> SolverTestHelper:
    # infeasible boolean problem - https://trac.sagemath.org/ticket/31962#comment:48
    z = cp.Variable(11, boolean=True)
    constraints = [z[2] + z[1] == 1,
                   z[4] + z[3] == 1,
                   z[6] + z[5] == 1,
                   z[8] + z[7] == 1,
                   z[10] + z[9] == 1,
                   z[4] + z[1] <= 1,
                   z[2] + z[3] <= 1,
                   z[6] + z[2] <= 1,
                   z[1] + z[5] <= 1,
                   z[8] + z[6] <= 1,
                   z[5] + z[7] <= 1,
                   z[10] + z[8] <= 1,
                   z[7] + z[9] <= 1,
                   z[9] + z[4] <= 1,
                   z[3] + z[10] <= 1]
    obj = cp.Minimize(0)
    obj_pair = (obj, np.inf)
    con_pairs = [(c, None) for c in constraints]
    var_pairs = [(z, None)]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth

def mi_lp_6() -> SolverTestHelper:
    "Test MILP for timelimit and no feasible solution"
    n = 70
    m = 70
    x = cp.Variable((n,), boolean=True, name="x")
    y = cp.Variable((n,), name="y")
    z = cp.Variable((m,), pos=True, name="z")
    A = np.random.rand(m, n)
    b = np.random.rand(m)
    objective = cp.Maximize(cp.sum(y))
    constraints = [
        A @ y <= b,
        y <= 1,
        cp.sum(x) >= 10,
        cp.sum(x) <= 20,
        z[0] + z[1] + z[2] >= 10,
        z[3] + z[4] + z[5] >= 5,
        z[6] + z[7] + z[8] >= 7,
        z[9] + z[10] >= 8,
        z[11] + z[12] >= 6,
        z[13] + z[14] >= 3,
        z[15] + z[16] >= 2,
        z[17] + z[18] >= 1,
        z[19] >= 2,
        z[20] >= 1,
        z[21] >= 1,
        z[22] >= 1,
        z[23] >= 1,
        z[24] >= 1,
        z[25] >= 1,
        z[26] >= 1,
        z[27] >= 1,
        z[28] >= 1,
        z[29] >= 1,
    ]
    return SolverTestHelper(
        (objective, None),
        [(x, None), (y, None), (z, None)],
        [(con, None) for con in constraints]
    )



def mi_lp_7() -> SolverTestHelper:
    """Problem that takes significant time to solve - for testing time/iteration limits"""
    np.random.seed(0)
    n = 24 * 8
    c = cp.Variable((n,), pos=True)
    d = cp.Variable((n,), pos=True)
    c_or_d = cp.Variable((n,), boolean=True)
    big = 1e3
    s = cp.cumsum(c * 0.9 - d)
    p = np.random.random(n)
    objective = cp.Maximize(p @ (d - c))
    constraints = [
        d <= 1,
        c <= 1,
        s >= 0,
        s <= 1,
        c <= c_or_d * big,
        d <= (1 - c_or_d) * big,
    ]
    return SolverTestHelper(
        (objective, None),
        [(c, None,), (d, None), (c_or_d, None)],
        [(con, None) for con in constraints]
    )


def mi_socp_1() -> SolverTestHelper:
    """
    Formulate the following mixed-integer SOCP with cvxpy
        min 3 * x[0] + 2 * x[1] + x[2] +  y[0] + 2 * y[1]
        s.t. norm(x,2) <= y[0]
             norm(x,2) <= y[1]
             x[0] + x[1] + 3*x[2] >= 0.1
             y <= 5, y integer.
    """
    x = cp.Variable(shape=(3,))
    y = cp.Variable(shape=(2,), integer=True)
    constraints = [cp.norm(x, 2) <= y[0],
                   cp.norm(x, 2) <= y[1],
                   x[0] + x[1] + 3 * x[2] >= 0.1,
                   y <= 5]
    obj = cp.Minimize(3 * x[0] + 2 * x[1] + x[2] + y[0] + 2 * y[1])
    obj_pair = (obj, 0.21363997604807272)
    var_pairs = [(x, np.array([-0.78510265, -0.43565177,  0.44025147])),
                 (y, np.array([1, 1]))]
    con_pairs = [(c, None) for c in constraints]  # no dual values for mixed-integer problems.
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def mi_socp_2() -> SolverTestHelper:
    """
    An (unnecessarily) SOCP-based reformulation of MI_LP_1.
    Doesn't use SOC objects.
    """
    x = cp.Variable(shape=(2,))
    bool_var = cp.Variable(boolean=True)
    int_var = cp.Variable(integer=True)
    objective = cp.Minimize(-4 * x[0] - 5 * x[1])
    constraints = [2 * x[0] + x[1] <= int_var,
                   (x[0] + 2 * x[1]) ** 2 <= 9 * bool_var,
                   x >= 0,
                   int_var == 3 * bool_var,
                   int_var == 3]
    obj_pair = (objective, -9)
    var_pairs = [(x, np.array([1, 1])),
                 (bool_var, 1),
                 (int_var, 3)]
    con_pairs = [(con, None) for con in constraints]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def mi_pcp_0() -> SolverTestHelper:
    """
    max  x3 + x4 - x0
    s.t. x0 + x1 + x2 / 2 == 2,
         (x0, x1, x3) in Pow3D(0.2)
         (x2, q, x4) in Pow3D(0.4)
         0.1 <= q <= 1.9,
         q integer
    """
    x = cp.Variable(shape=(3,))
    hypos = cp.Variable(shape=(2,))
    q = cp.Variable(integer=True)
    objective = cp.Minimize(-cp.sum(hypos) + x[0])
    arg1 = cp.hstack([x[0], x[2]])
    arg2 = cp.hstack(([x[1], q]))
    pc_con = cp.constraints.PowCone3D(arg1, arg2, hypos, [0.2, 0.4])
    con_pairs = [
        (x[0] + x[1] + 0.5 * x[2] == 2, None),
        (pc_con, None),
        (0.1 <= q, None),
        (q <= 1.9, None)
    ]
    obj_pair = (objective, -1.8073406786220672)
    var_pairs = [
        (x, np.array([0.06393515, 0.78320961, 2.30571048])),
        (hypos, None),
        (q, 1.0)
    ]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


class StandardTestLPs:

    @staticmethod
    def test_lp_0(solver, places: int = 4, duals: bool = True, **kwargs) -> SolverTestHelper:
        sth = lp_0()
        sth.solve(solver, **kwargs)
        sth.verify_primal_values(places)
        sth.verify_objective(places)
        if duals:
            sth.check_complementarity(places)
        return sth

    @staticmethod
    def test_lp_1(solver, places: int = 4, duals: bool = True, **kwargs) -> SolverTestHelper:
        sth = lp_1()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        if duals:
            sth.verify_dual_values(places)
        return sth

    @staticmethod
    def test_lp_2(solver, places: int = 4, duals: bool = True, **kwargs) -> SolverTestHelper:
        sth = lp_2()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        if duals:
            sth.verify_dual_values(places)
        return sth

    @staticmethod
    def test_lp_3(solver, places: int = 4, **kwargs) -> SolverTestHelper:
        sth = lp_3()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        return sth

    @staticmethod
    def test_lp_4(solver, places: int = 4, **kwargs) -> SolverTestHelper:
        sth = lp_4()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        return sth

    @staticmethod
    def test_lp_5(solver, places: int = 4, duals: bool = True,  **kwargs) -> SolverTestHelper:
        sth = lp_5()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.check_primal_feasibility(places)
        if duals:
            sth.check_complementarity(places)
            sth.check_dual_domains(places)
        return sth

    @staticmethod
    def test_lp_6(solver, places: int = 4, duals: bool = True, **kwargs) -> SolverTestHelper:
        sth = lp_6()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.check_primal_feasibility(places)
        if duals:
            sth.check_complementarity(places)
            sth.check_dual_domains(places)
        return sth

    @staticmethod
    def test_lp_7(solver, places: int = 4, duals: bool = True, **kwargs) -> SolverTestHelper:
        sth = lp_7()
        import sdpap
        if sdpap.sdpacall.sdpacall.get_backend_info()["gmp"]:
            sth.solve(solver, **kwargs)
            sth.verify_objective(places)
        return sth

    @staticmethod
    def test_mi_lp_0(solver, places: int = 4, **kwargs) -> SolverTestHelper:
        sth = mi_lp_0()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        return sth

    @staticmethod
    def test_mi_lp_1(solver, places: int = 4, **kwargs) -> SolverTestHelper:
        sth = mi_lp_1()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        return sth

    @staticmethod
    def test_mi_lp_2(solver, places: int = 4, **kwargs) -> SolverTestHelper:
        sth = mi_lp_2()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        return sth

    @staticmethod
    def test_mi_lp_3(solver, places: int = 4, **kwargs) -> SolverTestHelper:
        sth = mi_lp_3()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        return sth

    @staticmethod
    def test_mi_lp_4(solver, places: int = 4, **kwargs) -> SolverTestHelper:
        sth = mi_lp_4()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        return sth

    @staticmethod
    def test_mi_lp_5(solver, places: int = 4, **kwargs) -> SolverTestHelper:
        sth = mi_lp_5()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        return sth


class StandardTestQPs:

    @staticmethod
    def test_qp_0(solver, places: int = 4, duals: bool = True, **kwargs) -> SolverTestHelper:
        sth = qp_0()
        sth.solve(solver, **kwargs)
        sth.verify_primal_values(places)
        sth.verify_objective(places)
        if duals:
            sth.check_complementarity(places)
            sth.verify_dual_values(places)
        return sth


class StandardTestSOCPs:

    @staticmethod
    def test_socp_0(solver, places: int = 4, duals: bool = True, **kwargs) -> SolverTestHelper:
        sth = socp_0()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        if duals:
            sth.check_complementarity(places)
        return sth

    @staticmethod
    def test_socp_1(solver, places: int = 4, duals: bool = True, **kwargs) -> SolverTestHelper:
        sth = socp_1()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        if duals:
            sth.check_complementarity(places)
            sth.verify_dual_values(places)
        return sth

    @staticmethod
    def test_socp_2(solver, places: int = 4, duals: bool = True, **kwargs) -> SolverTestHelper:
        sth = socp_2()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        if duals:
            sth.check_complementarity(places)
            sth.verify_dual_values(places)
        return sth

    @staticmethod
    def test_socp_3ax0(solver, places: int = 3, duals: bool = True, **kwargs) -> SolverTestHelper:
        sth = socp_3(axis=0)
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        if duals:
            sth.check_complementarity(places)
            sth.verify_dual_values(places)
        return sth

    @staticmethod
    def test_socp_3ax1(solver, places: int = 3, duals: bool = True, **kwargs) -> SolverTestHelper:
        sth = socp_3(axis=1)
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        if duals:
            sth.check_complementarity(places)
            sth.verify_dual_values(places)
        return sth

    @staticmethod
    def test_mi_socp_1(solver, places: int = 4, **kwargs) -> SolverTestHelper:
        sth = mi_socp_1()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        return sth

    @staticmethod
    def test_mi_socp_2(solver, places: int = 4, **kwargs) -> SolverTestHelper:
        sth = mi_socp_2()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        return sth


class StandardTestSDPs:

    @staticmethod
    def test_sdp_1min(solver, places: int = 4, duals: bool = True, **kwargs) -> SolverTestHelper:
        sth = sdp_1('min')
        sth.solve(solver, **kwargs)
        sth.verify_objective(places=2)  # only 2 digits recorded.
        sth.check_primal_feasibility(places)
        if duals:
            sth.check_complementarity(places)
            sth.check_dual_domains(places)
        return sth

    @staticmethod
    def test_sdp_1max(solver, places: int = 4, duals: bool = True, **kwargs) -> SolverTestHelper:
        sth = sdp_1('max')
        sth.solve(solver, **kwargs)
        sth.verify_objective(places=2)  # only 2 digits recorded.
        sth.check_primal_feasibility(places)
        if duals:
            sth.check_complementarity(places)
            sth.check_dual_domains(places)
        return sth

    @staticmethod
    def test_sdp_2(solver, places: int = 2, duals: bool = True, **kwargs) -> SolverTestHelper:
        # places is set to 3 rather than 4, because analytic solution isn't known.
        sth = sdp_2()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.check_primal_feasibility(places)
        sth.verify_primal_values(places)
        if duals:
            sth.check_complementarity(places)
            sth.check_dual_domains(places)
        return sth


class StandardTestECPs:

    @staticmethod
    def test_expcone_1(solver, places: int = 4, duals: bool = True, **kwargs) -> SolverTestHelper:
        sth = expcone_1()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        if duals:
            sth.check_complementarity(places)
            sth.verify_dual_values(places)
        return sth


class StandardTestMixedCPs:

    @staticmethod
    def test_exp_soc_1(solver, places: int = 3, duals: bool = True, **kwargs) -> SolverTestHelper:
        sth = expcone_socp_1()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        if duals:
            sth.check_complementarity(places)
            sth.verify_dual_values(places)
        return sth

    @staticmethod
    def test_sdp_pcp_1(solver, places: int = 3, duals: bool = False, **kwargs) -> SolverTestHelper:
        sth = sdp_pcp_1()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.check_primal_feasibility(places)
        sth.verify_primal_values(places)
        if duals:
            sth.check_complementarity(places)
            sth.check_dual_domains(places)
        return sth

    
class StandardTestPCPs:

    @staticmethod
    def test_pcp_1(solver, places: int = 3, duals: bool = True, **kwargs) -> SolverTestHelper:
        sth = pcp_1()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.check_primal_feasibility(places)
        sth.verify_primal_values(places)
        if duals:
            sth.check_complementarity(places)
            sth.verify_dual_values(places)
        return sth

    @staticmethod
    def test_pcp_2(solver, places: int = 3, duals: bool = True, **kwargs) -> SolverTestHelper:
        sth = pcp_2()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.check_primal_feasibility(places)
        sth.verify_primal_values(places)
        if duals:
            sth.check_complementarity(places)
            sth.verify_dual_values(places)
        return sth

    @staticmethod
    def test_pcp_3(solver, places: int = 3, duals: bool = True, **kwargs) -> SolverTestHelper:
        sth = pcp_3()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.check_primal_feasibility(places)
        sth.verify_primal_values(places)
        if duals:
            sth.check_complementarity(places)
            sth.verify_dual_values(places)

    @staticmethod
    def test_mi_pcp_0(solver, places: int = 3, **kwargs) -> SolverTestHelper:
        sth = mi_pcp_0()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.check_primal_feasibility(places)
        sth.verify_primal_values(places)
        return sth
