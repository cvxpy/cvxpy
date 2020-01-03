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
import numpy as np
import cvxpy as cvx


class SolverTestHelper(object):

    def __init__(self, obj_pair, var_pairs, con_pairs):
        self.objective = obj_pair[0]
        self.constraints = [c for c, _ in con_pairs]
        self.prob = cvx.Problem(self.objective, self.constraints)
        self.variables = [x for x, _ in var_pairs]

        self.expect_val = obj_pair[1]
        self.expect_dual_vars = [dv for _, dv in con_pairs]
        self.expect_prim_vars = [pv for _, pv in var_pairs]

    def solve(self, solver, **kwargs):
        self.prob.solve(solver=solver, **kwargs)

    def check_objective(self, places=4):
        actual = self.prob.objective.value
        expect = self.prob.value
        assert abs(actual - expect) <= 10**(-places)

    def check_primal_feasibility(self, places):
        for con in self.constraints:
            viol = con.violation()
            if isinstance(viol, np.ndarray):
                viol = np.linalg.norm(viol, ord=2)
            assert viol <= 10**(-places)
        pass

    def check_dual_domains(self, places):
        # A full "dual feasibility" check would involve checking a stationary Lagrangian.
        # No such test is planned here.
        for con in self.constraints:
            if isinstance(con, cvx.constraints.PSD):
                dv = con.dual_value
                eigs = np.linalg.eigvalsh(dv)
                assert np.all(eigs >= -10**(-places))
            elif isinstance(con, cvx.constraints.ExpCone):
                raise NotImplementedError()
            elif isinstance(con, cvx.constraints.SOC):
                raise NotImplementedError()
            elif isinstance(con, cvx.constraints.Inequality):
                dv = con.dual_value
                assert np.all(dv >= -10**(-places))
            # for equality constraints, there is nothing to check
        pass

    def check_complementarity(self, places):
        for con in self.constraints:
            if isinstance(con, cvx.constraints.PSD):
                dv = con.dual_value
                pv = con.args[0].value
                comp = cvx.scalar_product(pv, dv).value
            elif isinstance(con, (cvx.constraints.ExpCone,
                                  cvx.constraints.SOC,
                                  cvx.constraints.NonPos,
                                  cvx.constraints.Zero)):
                comp = cvx.scalar_product(con.args, con.dual_value).value
            elif isinstance(con, (cvx.constraints.Inequality,
                                  cvx.constraints.Equality)):
                comp = cvx.scalar_product(con.expr, con.dual_value).value
            else:
                raise RuntimeError('Unknown constraint type.')
            assert abs(comp) <= 10**(-places)
        pass

    def verify_objective(self, places):
        actual = self.prob.value
        expect = self.expect_val
        if expect is not None:
            assert abs(actual - expect) < 10**(-places)

    def verify_primal_values(self, places):
        for idx in range(len(self.variables)):
            actual = self.variables[idx].value
            expect = self.expect_prim_vars[idx]
            if expect is not None:
                assert np.allclose(actual, expect, rtol=0, atol=10**(-places))
        pass

    def verify_dual_values(self, places):
        for idx in range(len(self.constraints)):
            actual = self.constraints[idx].dual_value
            expect = self.expect_dual_vars[idx]
            if expect is not None:
                if isinstance(actual, list):
                    for i in range(len(actual)):
                        act = actual[i]
                        exp = expect[i]
                        assert np.allclose(act, exp, rtol=0, atol=10**(-places))
                else:
                    assert np.allclose(actual, expect, rtol=0, atol=10**(-places))
        pass


def lp_0():
    x = cvx.Variable(shape=(2,))
    con_pairs = [(x == 0, None)]
    obj_pair = (cvx.Minimize(cvx.norm(x, 1) + 1.0), 1)
    var_pairs = [(x, np.array([0, 0]))]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def lp_1():
    # Example from
    # http://cvxopt.org/userguide/coneprog.html?highlight=solvers.lp#cvxopt.solvers.lp
    x = cvx.Variable(shape=(2,), name='x')
    objective = cvx.Minimize(-4 * x[0] - 5 * x[1])
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


def lp_2():
    x = cvx.Variable(shape=(2,), name='x')
    objective = cvx.Minimize(x[0] + 0.5 * x[1])
    constraints = [x[0] >= -100, x[0] <= -10, x[1] == 1]
    con_pairs = [(constraints[0], 1),
                 (constraints[1], 0),
                 (constraints[2], -0.5)]
    var_pairs = [(x, np.array([-100, 1]))]
    obj_pair = (objective, -99.5)
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def socp_0():
    x = cvx.Variable(shape=(2,))
    obj_pair = (cvx.Minimize(cvx.norm(x, 2) + 1), 1)
    con_pairs = [(x == 0, None)]
    var_pairs = [(x, np.array([0, 0]))]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def socp_1():
    """
    min 3 * x[0] + 2 * x[1] + x[2]
    s.t. norm(x,2) <= y
         x[0] + x[1] + 3*x[2] >= 1.0
         y <= 5
    """
    x = cvx.Variable(shape=(3,))
    y = cvx.Variable()
    soc = cvx.constraints.second_order.SOC(y, x)
    constraints = [soc,
                   x[0] + x[1] + 3 * x[2] >= 1.0,
                   y <= 5]
    obj = cvx.Minimize(3 * x[0] + 2 * x[1] + x[2])
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


def socp_2():
    """
    An (unnecessarily) SOCP-based reformulation of LP_1.
    Doesn't use SOC objects.
    """
    x = cvx.Variable(shape=(2,), name='x')
    objective = cvx.Minimize(-4 * x[0] - 5 * x[1])
    constraints = [2 * x[0] + x[1] <= 3,
                   (x[0] + 2 * x[1])**2 <= 3**2,
                   x[0] >= 0,
                   x[1] >= 0]
    con_pairs = [(constraints[0], 1),
                 (constraints[1], 1.0/3.0),
                 (constraints[2], 0),
                 (constraints[3], 0)]
    var_pairs = [(x, np.array([1, 1]))]
    obj_pair = (objective, -9)
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def sdp_1(objective_sense):
    """
    Solve "Example 8.3" from Convex Optimization by Boyd & Vandenberghe.

    Verify (1) optimal objective values, (2) that the dual variable to the PSD constraint
    belongs to the correct cone (i.e. the dual variable is itself PSD), and (3) that
    complementary slackness holds with the PSD primal variable and its dual variable.
    """
    rho = cvx.Variable(shape=(4, 4), symmetric=True)
    constraints = [0.6 <= rho[0, 1], rho[0, 1] <= 0.9,
                   0.8 <= rho[0, 2], rho[0, 2] <= 0.9,
                   0.5 <= rho[1, 3], rho[1, 3] <= 0.7,
                   -0.8 <= rho[2, 3], rho[2, 3] <= -0.4,
                   rho[0, 0] == 1, rho[1, 1] == 1, rho[2, 2] == 1, rho[3, 3] == 1,
                   rho >> 0]
    if objective_sense == 'min':
        obj = cvx.Minimize(rho[0, 3])
        obj_pair = (obj, -0.39)
    elif objective_sense == 'max':
        obj = cvx.Maximize(rho[0, 3])
        obj_pair = (obj, 0.23)
    else:
        raise RuntimeError('Unknown objective_sense.')
    con_pairs = [(c, None) for c in constraints]
    var_pairs = [(rho, None)]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def expcone_1():
    """
    min   3 * x[0] + 2 * x[1] + x[2]
    s.t.  0.1 <= x[0] + x[1] + x[2] <= 1
          x >= 0
          x[0] >= x[1] * exp(x[2] / x[1])
    """
    x = cvx.Variable(shape=(3, 1))
    cone_con = cvx.constraints.ExpCone(x[2], x[1], x[0])
    constraints = [cvx.sum(x) <= 1.0,
                   cvx.sum(x) >= 0.1,
                   x >= 0,
                   cone_con]
    obj = cvx.Minimize(3 * x[0] + 2 * x[1] + x[2])
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


def mi_lp_0():
    x = cvx.Variable(shape=(2,))
    bool_var = cvx.Variable(boolean=True)
    con_pairs = [(x == bool_var, None),
                 (bool_var == 0, None)]
    obj_pair = (cvx.Minimize(cvx.norm(x, 1) + 1.0), 1)
    var_pairs = [(x, np.array([0, 0])),
                 (bool_var, 0)]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def mi_lp_1():
    x = cvx.Variable(2, name='x')
    boolvar = cvx.Variable(boolean=True)
    intvar = cvx.Variable(integer=True)
    objective = cvx.Minimize(-4 * x[0] - 5 * x[1])
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


def mi_lp_2():
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
    X = cvx.Variable(n, boolean=True)
    objective = cvx.Maximize(cvx.sum(cvx.multiply([i[1] for i in coeffs], X)))
    constraints = [cvx.sum(cvx.multiply([i[2] for i in coeffs], X)) <= c]
    obj_pair = (objective, z)
    con_pairs = [(constraints[0], None)]
    var_pairs = [(X, None)]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def mi_socp_1():
    """
    Formulate the following mixed-integer SOCP with cvxpy
        min 3 * x[0] + 2 * x[1] + x[2] +  y[0] + 2 * y[1]
        s.t. norm(x,2) <= y[0]
             norm(x,2) <= y[1]
             x[0] + x[1] + 3*x[2] >= 0.1
             y <= 5, y integer.
    and solve with MOSEK.
    """
    x = cvx.Variable(shape=(3,))
    y = cvx.Variable(shape=(2,), integer=True)
    constraints = [cvx.norm(x, 2) <= y[0],
                   cvx.norm(x, 2) <= y[1],
                   x[0] + x[1] + 3 * x[2] >= 0.1,
                   y <= 5]
    obj = cvx.Minimize(3 * x[0] + 2 * x[1] + x[2] + y[0] + 2 * y[1])
    obj_pair = (obj, 0.21363997604807272)
    var_pairs = [(x, np.array([-0.78510265, -0.43565177,  0.44025147])),
                 (y, np.array([1, 1]))]
    con_pairs = [(c, None) for c in constraints]  # no dual values for mixed-integer problems.
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth


def mi_socp_2():
    """
    An (unnecessarily) SOCP-based reformulation of MI_LP_1.
    Doesn't use SOC objects.
    """
    x = cvx.Variable(shape=(2,))
    bool_var = cvx.Variable(boolean=True)
    int_var = cvx.Variable(integer=True)
    objective = cvx.Minimize(-4 * x[0] - 5 * x[1])
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


class StandardTestLPs(object):

    @staticmethod
    def test_lp_0(solver, places=4, **kwargs):
        sth = lp_0()
        sth.solve(solver, **kwargs)
        sth.verify_primal_values(places)
        sth.verify_objective(places)
        sth.check_complementarity(places)

    @staticmethod
    def test_lp_1(solver, places=4, **kwargs):
        sth = lp_1()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        sth.verify_dual_values(places)

    @staticmethod
    def test_lp_2(solver, places=4, **kwargs):
        sth = lp_2()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        sth.verify_dual_values(places)

    @staticmethod
    def test_mi_lp_0(solver, places=4, **kwargs):
        sth = mi_lp_0()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)

    @staticmethod
    def test_mi_lp_1(solver, places=4, **kwargs):
        sth = mi_lp_1()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)

    @staticmethod
    def test_mi_lp_2(solver, places=4, **kwargs):
        sth = mi_lp_2()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)


class StandardTestSOCPs(object):
    
    @staticmethod
    def test_socp_0(solver, places=4, **kwargs):
        sth = socp_0()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        sth.check_complementarity(places)
        pass
    
    @staticmethod
    def test_socp_1(solver, places=4, **kwargs):
        sth = socp_1()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        sth.check_complementarity(places)
        sth.verify_dual_values(places)
        pass
    
    @staticmethod
    def test_socp_2(solver, places=4, **kwargs):
        sth = socp_2()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        sth.check_complementarity(places)
        sth.verify_dual_values(places)
        pass
    
    @staticmethod
    def test_mi_socp_1(solver, places=4, **kwargs):
        sth = mi_socp_1()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        pass
        
    @staticmethod
    def test_mi_socp_2(solver, places=4, **kwargs):
        sth = mi_socp_2()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        pass
    
    
class StandardTestSDPs(object):
    
    @staticmethod
    def test_sdp_1min(solver, places=4, **kwargs):
        sth = sdp_1('min')
        sth.solve(solver, **kwargs)
        sth.verify_objective(places=2)  # only 2 digits recorded.
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        pass

    @staticmethod
    def test_sdp_1max(solver, places=4, **kwargs):
        sth = sdp_1('max')
        sth.solve(solver, **kwargs)
        sth.verify_objective(places=2)  # only 2 digits recorded.
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        pass


class StandardTestECPs(object):
    
    @staticmethod
    def test_expcone_1(solver, places=4, **kwargs):
        sth = expcone_1()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_dual_values(places)
        sth.verify_primal_values(places)
        pass
