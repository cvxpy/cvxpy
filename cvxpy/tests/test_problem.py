"""
Copyright 2017 Steven Diamond

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

from fractions import Fraction
import cvxpy.settings as s
import cvxpy as cp
from cvxpy.constraints import NonPos, Zero, ExpCone, PSD
from cvxpy.error import DCPError, ParameterError, SolverError
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.problems.problem import Problem
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.conic_solvers import ecos_conif, scs_conif
from cvxpy.reductions.solvers.defines import SOLVER_MAP_CONIC, INSTALLED_SOLVERS
import cvxpy.interface as intf
from cvxpy.tests.base_test import BaseTest
from numpy import linalg as LA
import numpy
import numpy as np
import scipy.sparse as sp
import builtins
import sys
import pickle
# Solvers.
import scs
import ecos
import warnings
from io import StringIO


class TestProblem(BaseTest):
    """Unit tests for the expression/expression module.
    """

    def setUp(self) -> None:
        self.a = Variable(name='a')
        self.b = Variable(name='b')
        self.c = Variable(name='c')

        self.x = Variable(2, name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(2, name='z')

        self.A = Variable((2, 2), name='A')
        self.B = Variable((2, 2), name='B')
        self.C = Variable((3, 2), name='C')

    def test_to_str(self) -> None:
        """Test string representations.
        """
        obj = cp.Minimize(self.a)
        prob = Problem(obj)
        self.assertEqual(repr(prob), "Problem(%s, %s)" % (repr(obj), repr([])))
        constraints = [self.x*2 == self.x, self.x == 0]
        prob = Problem(obj, constraints)
        self.assertEqual(repr(prob), "Problem(%s, %s)" % (repr(obj), repr(constraints)))

        # Test str.
        result = (
            "minimize %(name)s\nsubject to %(name)s == 0\n           %(name)s <= 0" % {
                "name": self.a.name()
            }
        )
        prob = Problem(cp.Minimize(self.a), [Zero(self.a), NonPos(self.a)])
        self.assertEqual(str(prob), result)

    def test_variables(self) -> None:
        """Test the variables method.
        """
        p = Problem(cp.Minimize(self.a), [self.a <= self.x, self.b <= self.A + 2])
        vars_ = p.variables()
        ref = [self.a, self.x, self.b, self.A]
        self.assertCountEqual(vars_, ref)

    def test_var_dict(self) -> None:
        p = Problem(cp.Minimize(self.a), [self.a <= self.x, self.b <= self.A + 2])
        assert p.var_dict == {"a": self.a, "x": self.x, "b": self.b, "A": self.A}

    def test_parameters(self) -> None:
        """Test the parameters method.
        """
        p1 = Parameter()
        p2 = Parameter(3, nonpos=True)
        p3 = Parameter((4, 4), nonneg=True)
        p = Problem(cp.Minimize(p1), [self.a + p1 <= p2, self.b <= p3 + p3 + 2])
        params = p.parameters()
        ref = [p1, p2, p3]
        self.assertCountEqual(params, ref)

    def test_param_dict(self) -> None:
        p1 = Parameter(name="p1")
        p2 = Parameter(3, nonpos=True, name="p2")
        p3 = Parameter((4, 4), nonneg=True, name="p3")
        p = Problem(cp.Minimize(p1), [self.a + p1 <= p2, self.b <= p3 + p3 + 2])
        assert p.param_dict == {"p1": p1, "p2": p2, "p3": p3}

    def test_solving_a_problem_with_unspecified_parameters(self) -> None:
        param = cp.Parameter(name="lambda")
        problem = cp.Problem(cp.Minimize(param), [])
        with self.assertRaises(
              ParameterError, msg="A Parameter (whose name is 'lambda').*"):
            problem.solve()

    def test_constants(self) -> None:
        """Test the constants method.
        """
        c1 = numpy.random.randn(1, 2)
        c2 = numpy.random.randn(2)
        p = Problem(cp.Minimize(c1 @ self.x), [self.x >= c2])
        constants_ = p.constants()
        ref = [c1, c2]
        self.assertEqual(len(ref), len(constants_))
        for c, r in zip(constants_, ref):
            self.assertTupleEqual(c.shape, r.shape)
            self.assertTrue((c.value == r).all())
            # Allows comparison between numpy matrices and numpy arrays
            # Necessary because of the way cvxpy handles numpy arrays and constants

        # Single scalar constants
        p = Problem(cp.Minimize(self.a), [self.x >= 1])
        constants_ = p.constants()
        ref = [numpy.array(1)]
        self.assertEqual(len(ref), len(constants_))
        for c, r in zip(constants_, ref):
            self.assertEqual(c.shape, r.shape) and \
                self.assertTrue((c.value == r).all())
            # Allows comparison between numpy matrices and numpy arrays
            # Necessary because of the way cvxpy handles numpy arrays and constants

    def test_size_metrics(self) -> None:
        """Test the size_metrics method.
        """
        p1 = Parameter()
        p2 = Parameter(3, nonpos=True)
        p3 = Parameter((4, 4), nonneg=True)

        c1 = numpy.random.randn(2, 1)
        c2 = numpy.random.randn(1, 2)
        constants = [2, c2.dot(c1)]

        p = Problem(cp.Minimize(p1), [self.a + p1 <= p2,
                                      self.b <= p3 + p3 + constants[0],
                                      self.c == constants[1]])
        # num_scalar_variables
        n_variables = p.size_metrics.num_scalar_variables
        ref = self.a.size + self.b.size + self.c.size
        self.assertEqual(n_variables, ref)

        # num_scalar_data
        n_data = p.size_metrics.num_scalar_data
        # 2 and c2.dot(c1) are both single scalar constants.
        ref = numpy.prod(p1.size) + numpy.prod(p2.size) + numpy.prod(p3.size) + len(constants)
        self.assertEqual(n_data, ref)

        # num_scalar_eq_constr
        n_eq_constr = p.size_metrics.num_scalar_eq_constr
        ref = c2.dot(c1).size
        self.assertEqual(n_eq_constr, ref)

        # num_scalar_leq_constr
        n_leq_constr = p.size_metrics.num_scalar_leq_constr
        ref = numpy.prod(p3.size) + numpy.prod(p2.size)
        self.assertEqual(n_leq_constr, ref)

        # max_data_dimension
        max_data_dim = p.size_metrics.max_data_dimension
        ref = max(p3.shape)
        self.assertEqual(max_data_dim, ref)

    def test_solver_stats(self) -> None:
        """Test the solver_stats method.
        """
        prob = Problem(cp.Minimize(cp.norm(self.x)), [self.x == 0])
        prob.solve(solver=s.ECOS)
        stats = prob.solver_stats
        self.assertGreater(stats.solve_time, 0)
        self.assertGreater(stats.setup_time, 0)
        self.assertGreater(stats.num_iters, 0)
        self.assertIn('info', stats.extra_stats)

        prob = Problem(cp.Minimize(cp.norm(self.x)), [self.x == 0])
        prob.solve(solver=s.SCS)
        stats = prob.solver_stats
        self.assertGreater(stats.solve_time, 0)
        self.assertGreater(stats.setup_time, 0)
        self.assertGreater(stats.num_iters, 0)
        self.assertIn('info', stats.extra_stats)

        prob = Problem(cp.Minimize(cp.sum(self.x)), [self.x == 0])
        prob.solve(solver=s.OSQP)
        stats = prob.solver_stats
        self.assertGreater(stats.solve_time, 0)
        # We do not populate setup_time for OSQP (OSQP decomposes time
        # into setup, solve, and polish; these are summed to obtain solve_time)
        self.assertGreater(stats.num_iters, 0)
        self.assertTrue(hasattr(stats.extra_stats, 'info'))

    def test_get_problem_data(self) -> None:
        """Test get_problem_data method.
        """
        data, _, _ = Problem(cp.Minimize(cp.exp(self.a) + 2)).get_problem_data(s.SCS)
        dims = data[ConicSolver.DIMS]
        self.assertEqual(dims.exp, 1)
        self.assertEqual(data["c"].shape, (2,))
        self.assertEqual(data["A"].shape, (3, 2))

        data, _, _ = Problem(cp.Minimize(cp.norm(self.x) + 3)).get_problem_data(s.ECOS)
        dims = data[ConicSolver.DIMS]
        self.assertEqual(dims.soc, [3])
        self.assertEqual(data["c"].shape, (3,))
        self.assertIsNone(data["A"])
        self.assertEqual(data["G"].shape, (3, 3))

        if s.CVXOPT in INSTALLED_SOLVERS:
            data, _, _ = Problem(cp.Minimize(cp.norm(self.x) + 3)).get_problem_data(
                s.CVXOPT)
            dims = data[ConicSolver.DIMS]
            self.assertEqual(dims.soc, [3])
            # TODO(akshayka): We cannot test whether the coefficients or
            # offsets were correctly parsed until we update the CVXOPT
            # interface.

    def test_unpack_results(self) -> None:
        """Test unpack results method.
        """
        prob = Problem(cp.Minimize(cp.exp(self.a)), [self.a == 0])
        args, chain, inv = prob.get_problem_data(s.SCS)
        data = {"c": args["c"], "A": args["A"], "b": args["b"]}
        cones = scs_conif.dims_to_solver_dict(args[ConicSolver.DIMS])
        solution = scs.solve(data, cones)
        prob = Problem(cp.Minimize(cp.exp(self.a)), [self.a == 0])
        prob.unpack_results(solution, chain, inv)
        self.assertAlmostEqual(self.a.value, 0, places=3)
        self.assertAlmostEqual(prob.value, 1, places=3)
        self.assertAlmostEqual(prob.status, s.OPTIMAL)

        prob = Problem(cp.Minimize(cp.norm(self.x)), [self.x == 0])
        args, chain, inv = prob.get_problem_data(s.ECOS)
        cones = ecos_conif.dims_to_solver_dict(args[ConicSolver.DIMS])
        solution = ecos.solve(args["c"], args["G"], args["h"],
                              cones, args["A"], args["b"])
        prob = Problem(cp.Minimize(cp.norm(self.x)), [self.x == 0])
        prob.unpack_results(solution, chain, inv)
        self.assertItemsAlmostEqual(self.x.value, [0, 0])
        self.assertAlmostEqual(prob.value, 0)
        self.assertAlmostEqual(prob.status, s.OPTIMAL)

    def test_verbose(self) -> None:
        """Test silencing and enabling solver messages.
        """
        # From http://stackoverflow.com/questions/5136611/capture-stdout-from-a-script-in-python
        # setup the environment
        outputs = {True: [], False: []}
        backup = sys.stdout
        ######
        for solver in INSTALLED_SOLVERS:
            for verbose in [True, False]:
                # Don't test GLPK because there's a race
                # condition in setting CVXOPT solver options.
                if solver in [cp.GLPK, cp.GLPK_MI, cp.MOSEK, cp.CBC]:
                    continue
                sys.stdout = StringIO()  # capture output

                p = Problem(cp.Minimize(self.a + self.x[0]),
                            [self.a >= 2, self.x >= 2])
                p.solve(verbose=verbose, solver=solver)

                if solver in SOLVER_MAP_CONIC:
                    if SOLVER_MAP_CONIC[solver].MIP_CAPABLE:
                        p.constraints.append(Variable(boolean=True) == 0)
                        p.solve(verbose=verbose, solver=solver)

                    if ExpCone in SOLVER_MAP_CONIC[solver].SUPPORTED_CONSTRAINTS:
                        p = Problem(cp.Minimize(self.a), [cp.log(self.a) >= 2])
                        p.solve(verbose=verbose, solver=solver)

                    if PSD in SOLVER_MAP_CONIC[solver].SUPPORTED_CONSTRAINTS:
                        a_mat = cp.reshape(self.a, shape=(1, 1))
                        p = Problem(cp.Minimize(self.a), [cp.lambda_min(a_mat) >= 2])
                        p.solve(verbose=verbose, solver=solver)

                out = sys.stdout.getvalue()  # release output
                sys.stdout.close()  # close the stream
                sys.stdout = backup  # restore original stdout

                outputs[verbose].append((out, solver))

        for output, solver in outputs[True]:
            print(solver)
            assert len(output) > 0
        for output, solver in outputs[False]:
            print(solver)
            assert len(output) == 0

    # Test registering other solve methods.
    def test_register_solve(self) -> None:
        Problem.register_solve("test", lambda self: 1)
        p = Problem(cp.Minimize(1))
        result = p.solve(method="test")
        self.assertEqual(result, 1)

        def test(self, a, b: int = 2):
            return (a, b)
        Problem.register_solve("test", test)
        p = Problem(cp.Minimize(0))
        result = p.solve(1, b=3, method="test")
        self.assertEqual(result, (1, 3))
        result = p.solve(1, method="test")
        self.assertEqual(result, (1, 2))
        result = p.solve(1, method="test", b=4)
        self.assertEqual(result, (1, 4))

    # def test_consistency(self):
    #     """Test that variables and constraints keep a consistent order.
    #     """
    #     # TODO(akshayka): Adapt this test to the reduction infrastructure.
    #     import itertools
    #     num_solves = 4
    #     vars_lists = []
    #     ineqs_lists = []
    #     var_ids_order_created = []
    #     for k in range(num_solves):
    #         sum = 0
    #         constraints = []
    #         var_ids = []
    #         for i in range(100):
    #             var = Variable(name=str(i))
    #             var_ids.append(var.id)
    #             sum += var
    #             constraints.append(var >= i)
    #         var_ids_order_created.append(var_ids)
    #         obj = cp.Minimize(sum)
    #         p = Problem(obj, constraints)
    #         objective, constraints = p.canonicalize()
    #         sym_data = SymData(objective, constraints, SOLVERS[s.ECOS])
    #         # Sort by offset.
    #         vars_ = sorted(sym_data.var_offsets.items(),
    #                        key=lambda key_val: key_val[1])
    #         vars_ = [var_id for (var_id, offset) in vars_]
    #         vars_lists.append(vars_)
    #         ineqs_lists.append(sym_data.constr_map[s.LEQ])

    #     # Verify order of variables is consistent.
    #     for i in range(num_solves):
    #         self.assertEqual(var_ids_order_created[i],
    #                          vars_lists[i])
    #     for i in range(num_solves):
    #         for idx, constr in enumerate(ineqs_lists[i]):
    #             var_id, _ = lu.get_expr_vars(constr.expr)[0]
    #             self.assertEqual(var_ids_order_created[i][idx],
    #                              var_id)

    # Test removing duplicate constraint objects.
    # def test_duplicate_constraints(self):
    #     # TODO(akshayka): Adapt this test to the reduction infrastructure.
    #     eq = (self.x == 2)
    #     le = (self.x <= 2)
    #     obj = 0

    #     def test(self):
    #         objective, constraints = self.canonicalize()
    #         sym_data = SymData(objective, constraints, SOLVERS[s.CVXOPT])
    #         return (len(sym_data.constr_map[s.EQ]),
    #                 len(sym_data.constr_map[s.LEQ]))
    #     Problem.register_solve("test", test)
    #     p = Problem(cp.Minimize(obj), [eq, eq, le, le])
    #     result = p.solve(method="test")
    #     self.assertEqual(result, (1, 1))

    #     # Internal constraints.
    #     X = Variable((2, 2), PSD=True)
    #     obj = sum(X + X)
    #     p = Problem(cp.Minimize(obj))
    #     result = p.solve(method="test")
    #     self.assertEqual(result, (0, 1))

    #     # Duplicates from non-linear constraints.
    #     exp = norm(self.x, 2)
    #     prob = Problem(cp.Minimize(0), [exp <= 1, exp <= 2])
    #     result = prob.solve(method="test")
    #     self.assertEqual(result, (0, 3))

    # Test the is_dcp method.
    def test_is_dcp(self) -> None:
        p = Problem(cp.Minimize(cp.norm_inf(self.a)))
        self.assertEqual(p.is_dcp(), True)

        p = Problem(cp.Maximize(cp.norm_inf(self.a)))
        self.assertEqual(p.is_dcp(), False)
        with self.assertRaises(DCPError):
            p.solve()

    # Test the is_qp method.
    def test_is_qp(self) -> None:
        A = numpy.random.randn(4, 3)
        b = numpy.random.randn(4)
        Aeq = numpy.random.randn(2, 3)
        beq = numpy.random.randn(2)
        F = numpy.random.randn(2, 3)
        g = numpy.random.randn(2)
        obj = cp.sum_squares(A @ self.y - b)
        qpwa_obj = 3*cp.sum_squares(-cp.abs(A @ self.y)) +\
            cp.quad_over_lin(cp.maximum(cp.abs(A @ self.y), [3., 3., 3., 3.]), 2.)
        not_qpwa_obj = 3*cp.sum_squares(cp.abs(A @ self.y)) +\
            cp.quad_over_lin(cp.minimum(cp.abs(A @ self.y), [3., 3., 3., 3.]), 2.)

        p = Problem(cp.Minimize(obj), [])
        self.assertEqual(p.is_qp(), True)

        p = Problem(cp.Minimize(qpwa_obj), [])
        self.assertEqual(p.is_qp(), True)

        p = Problem(cp.Minimize(not_qpwa_obj), [])
        self.assertEqual(p.is_qp(), False)

        p = Problem(cp.Minimize(obj),
                    [Aeq @ self.y == beq, F @ self.y <= g])
        self.assertEqual(p.is_qp(), True)

        p = Problem(cp.Minimize(qpwa_obj),
                    [Aeq @ self.y == beq, F @ self.y <= g])
        self.assertEqual(p.is_qp(), True)

        p = Problem(cp.Minimize(obj), [cp.maximum(1, 3 * self.y) <= 200,
                                       cp.abs(2 * self.y) <= 100,
                                       cp.norm(2 * self.y, 1) <= 1000,
                                       Aeq @ self.y == beq])
        self.assertEqual(p.is_qp(), True)

        p = Problem(cp.Minimize(qpwa_obj), [cp.maximum(1, 3 * self.y) <= 200,
                                            cp.abs(2 * self.y) <= 100,
                                            cp.norm(2 * self.y, 1) <= 1000,
                                            Aeq @ self.y == beq])
        self.assertEqual(p.is_qp(), True)

        p = Problem(cp.Minimize(obj), [cp.maximum(1, 3 * self.y ** 2) <= 200])
        self.assertEqual(p.is_qp(), False)

        p = Problem(cp.Minimize(qpwa_obj), [cp.maximum(1, 3 * self.y ** 2) <= 200])
        self.assertEqual(p.is_qp(), False)

    # Test problems involving variables with the same name.
    def test_variable_name_conflict(self) -> None:
        var = Variable(name='a')
        p = Problem(cp.Maximize(self.a + var), [var == 2 + self.a, var <= 3])
        result = p.solve()
        self.assertAlmostEqual(result, 4.0)
        self.assertAlmostEqual(self.a.value, 1)
        self.assertAlmostEqual(var.value, 3)

    # Test adding problems
    def test_add_problems(self) -> None:
        prob1 = Problem(cp.Minimize(self.a), [self.a >= self.b])
        prob2 = Problem(cp.Minimize(2*self.b), [self.a >= 1, self.b >= 2])
        prob_minimize = prob1 + prob2
        self.assertEqual(len(prob_minimize.constraints), 3)
        self.assertAlmostEqual(prob_minimize.solve(), 6)
        prob3 = Problem(cp.Maximize(self.a), [self.b <= 1])
        prob4 = Problem(cp.Maximize(2*self.b), [self.a <= 2])
        prob_maximize = prob3 + prob4
        self.assertEqual(len(prob_maximize.constraints), 2)
        self.assertAlmostEqual(prob_maximize.solve(), 4)

        # Test using sum function
        prob5 = Problem(cp.Minimize(3*self.a))
        prob_sum = sum([prob1, prob2, prob5])
        self.assertEqual(len(prob_sum.constraints), 3)
        self.assertAlmostEqual(prob_sum.solve(), 12)
        prob_sum = sum([prob1])
        self.assertEqual(len(prob_sum.constraints), 1)

        # Test cp.Minimize + cp.Maximize
        with self.assertRaises(DCPError) as cm:
            prob1 + prob3
        self.assertEqual(str(cm.exception), "Problem does not follow DCP rules.")

    # Test problem multiplication by scalar
    def test_mul_problems(self) -> None:
        prob1 = Problem(cp.Minimize(pow(self.a, 2)), [self.a >= 2])
        answer = prob1.solve()
        factors = [0, 1, 2.3, -4.321]
        for f in factors:
            self.assertAlmostEqual((f * prob1).solve(), f * answer)
            self.assertAlmostEqual((prob1 * f).solve(), f * answer)

    # Test problem linear combinations
    def test_lin_combination_problems(self) -> None:
        prob1 = Problem(cp.Minimize(self.a), [self.a >= self.b])
        prob2 = Problem(cp.Minimize(2*self.b), [self.a >= 1, self.b >= 2])
        prob3 = Problem(cp.Maximize(-pow(self.b + self.a, 2)), [self.b >= 3])

        # simple addition and multiplication
        combo1 = prob1 + 2 * prob2
        combo1_ref = Problem(cp.Minimize(self.a + 4 * self.b),
                             [self.a >= self.b, self.a >= 1, self.b >= 2])
        self.assertAlmostEqual(combo1.solve(), combo1_ref.solve())

        # division and subtraction
        combo2 = prob1 - prob3/2
        combo2_ref = Problem(cp.Minimize(self.a + pow(self.b + self.a, 2)/2),
                             [self.b >= 3, self.a >= self.b])
        self.assertAlmostEqual(combo2.solve(), combo2_ref.solve())

        # multiplication with 0 (prob2's constraints should still hold)
        combo3 = prob1 + 0 * prob2 - 3 * prob3
        combo3_ref = Problem(cp.Minimize(self.a + 3 * pow(self.b + self.a, 2)),
                             [self.a >= self.b, self.a >= 1, self.b >= 3])
        self.assertAlmostEqual(combo3.solve(), combo3_ref.solve())

    # Test scalar LP problems.
    def test_scalar_lp(self) -> None:
        p = Problem(cp.Minimize(3*self.a), [self.a >= 2])
        result = p.solve()
        self.assertAlmostEqual(result, 6)
        self.assertAlmostEqual(self.a.value, 2)

        p = Problem(cp.Maximize(3*self.a - self.b),
                    [self.a <= 2, self.b == self.a, self.b <= 5])
        result = p.solve()
        self.assertAlmostEqual(result, 4.0)
        self.assertAlmostEqual(self.a.value, 2)
        self.assertAlmostEqual(self.b.value, 2)

        # With a constant in the objective.
        p = Problem(cp.Minimize(3*self.a - self.b + 100),
                    [self.a >= 2,
                     self.b + 5*self.c - 2 == self.a,
                     self.b <= 5 + self.c])
        result = p.solve()
        self.assertAlmostEqual(result, 101 + 1.0/6)
        self.assertAlmostEqual(self.a.value, 2)
        self.assertAlmostEqual(self.b.value, 5-1.0/6)
        self.assertAlmostEqual(self.c.value, -1.0/6)

        # Test status and value.
        exp = cp.Maximize(self.a)
        p = Problem(exp, [self.a <= 2])
        result = p.solve(solver=s.ECOS)
        self.assertEqual(result, p.value)
        self.assertEqual(p.status, s.OPTIMAL)
        assert self.a.value is not None
        assert p.constraints[0].dual_value is not None

        # Unbounded problems.
        p = Problem(cp.Maximize(self.a), [self.a >= 2])
        p.solve(solver=s.ECOS)
        self.assertEqual(p.status, s.UNBOUNDED)
        assert numpy.isinf(p.value)
        assert p.value > 0
        assert self.a.value is None
        assert p.constraints[0].dual_value is None

        if s.CVXOPT in INSTALLED_SOLVERS:
            p = Problem(cp.Minimize(-self.a), [self.a >= 2])
            result = p.solve(solver=s.CVXOPT)
            self.assertEqual(result, p.value)
            self.assertEqual(p.status, s.UNBOUNDED)
            assert numpy.isinf(p.value)
            assert p.value < 0

        # Infeasible problems.
        p = Problem(cp.Maximize(self.a), [self.a >= 2, self.a <= 1])
        self.a.save_value(2)
        p.constraints[0].save_dual_value(2)

        result = p.solve(solver=s.ECOS)
        self.assertEqual(result, p.value)
        self.assertEqual(p.status, s.INFEASIBLE)
        assert numpy.isinf(p.value)
        assert p.value < 0
        assert self.a.value is None
        assert p.constraints[0].dual_value is None

        p = Problem(cp.Minimize(-self.a), [self.a >= 2, self.a <= 1])
        result = p.solve(solver=s.ECOS)
        self.assertEqual(result, p.value)
        self.assertEqual(p.status, s.INFEASIBLE)
        assert numpy.isinf(p.value)
        assert p.value > 0

    # Test vector LP problems.
    def test_vector_lp(self) -> None:
        c = Constant(numpy.array([[1, 2]]).T).value
        p = Problem(cp.Minimize(c.T @ self.x), [self.x[:, None] >= c])
        result = p.solve()
        self.assertAlmostEqual(result, 5)
        self.assertItemsAlmostEqual(self.x.value, [1, 2])

        A = Constant(numpy.array([[3, 5], [1, 2]]).T).value
        Imat = Constant([[1, 0], [0, 1]])
        p = Problem(cp.Minimize(c.T @ self.x + self.a),
                    [A @ self.x >= [-1, 1],
                     4*Imat @ self.z == self.x,
                     self.z >= [2, 2],
                     self.a >= 2])
        result = p.solve()
        self.assertAlmostEqual(result, 26, places=3)
        obj = (c.T @ self.x + self.a).value[0]
        self.assertAlmostEqual(obj, result)
        self.assertItemsAlmostEqual(self.x.value, [8, 8], places=3)
        self.assertItemsAlmostEqual(self.z.value, [2, 2], places=3)

    def test_ecos_noineq(self) -> None:
        """Test ECOS with no inequality constraints.
        """
        T = Constant(numpy.ones((2, 2))).value
        p = Problem(cp.Minimize(1), [self.A == T])
        result = p.solve(solver=s.ECOS)
        self.assertAlmostEqual(result, 1)
        self.assertItemsAlmostEqual(self.A.value, T)

    # Test matrix LP problems.
    def test_matrix_lp(self) -> None:
        T = Constant(numpy.ones((2, 2))).value
        p = Problem(cp.Minimize(1), [self.A == T])
        result = p.solve()
        self.assertAlmostEqual(result, 1)
        self.assertItemsAlmostEqual(self.A.value, T)

        T = Constant(numpy.ones((2, 3))*2).value
        p = Problem(cp.Minimize(1), [self.A >= T @ self.C,
                                     self.A == self.B, self.C == T.T])
        result = p.solve(solver=cp.ECOS)
        self.assertAlmostEqual(result, 1)
        self.assertItemsAlmostEqual(self.A.value, self.B.value)
        self.assertItemsAlmostEqual(self.C.value, T)
        assert (self.A.value >= (T @ self.C).value).all()

        # Test variables are dense.
        self.assertEqual(type(self.A.value), intf.DEFAULT_INTF.TARGET_MATRIX)

    # Test variable promotion.
    def test_variable_promotion(self) -> None:
        p = Problem(cp.Minimize(self.a), [self.x <= self.a, self.x == [1, 2]])
        result = p.solve()
        self.assertAlmostEqual(result, 2)
        self.assertAlmostEqual(self.a.value, 2)

        p = Problem(cp.Minimize(self.a),
                    [self.A <= self.a,
                     self.A == [[1, 2], [3, 4]]
                     ])
        result = p.solve()
        self.assertAlmostEqual(result, 4)
        self.assertAlmostEqual(self.a.value, 4)

        # Promotion must happen before the multiplication.
        p = Problem(cp.Minimize([[1], [1]] @ (self.x + self.a + 1)),
                    [self.a + self.x >= [1, 2]])
        result = p.solve()
        self.assertAlmostEqual(result, 5)

    # Test parameter promotion.
    def test_parameter_promotion(self) -> None:
        a = Parameter()
        exp = [[1, 2], [3, 4]] * a
        a.value = 2
        assert not (exp.value - 2*numpy.array([[1, 2], [3, 4]]).T).any()

    def test_parameter_problems(self) -> None:
        """Test problems with parameters.
        """
        p1 = Parameter()
        p2 = Parameter(3, nonpos=True)
        p3 = Parameter((4, 4), nonneg=True)
        p = Problem(cp.Maximize(p1*self.a), [self.a + p1 <= p2, self.b <= p3 + p3 + 2])
        p1.value = 2
        p2.value = -numpy.ones((3,))
        p3.value = numpy.ones((4, 4))
        result = p.solve()
        self.assertAlmostEqual(result, -6)

        p1.value = None
        with self.assertRaises(ParameterError):
            p.solve()

    # Test problems with norm_inf
    def test_norm_inf(self) -> None:
        # Constant argument.
        p = Problem(cp.Minimize(cp.norm_inf(-2)))
        result = p.solve()
        self.assertAlmostEqual(result, 2)

        # Scalar arguments.
        p = Problem(cp.Minimize(cp.norm_inf(self.a)), [self.a >= 2])
        result = p.solve()
        self.assertAlmostEqual(result, 2)
        self.assertAlmostEqual(self.a.value, 2)

        p = Problem(cp.Minimize(3*cp.norm_inf(self.a + 2*self.b) + self.c),
                    [self.a >= 2, self.b <= -1, self.c == 3])
        result = p.solve()
        self.assertAlmostEqual(result, 3)
        self.assertAlmostEqual(self.a.value + 2*self.b.value, 0)
        self.assertAlmostEqual(self.c.value, 3)

        # cp.Maximize
        p = Problem(cp.Maximize(-cp.norm_inf(self.a)), [self.a <= -2])
        result = p.solve()
        self.assertAlmostEqual(result, -2)
        self.assertAlmostEqual(self.a.value, -2)

        # Vector arguments.
        p = Problem(cp.Minimize(cp.norm_inf(self.x - self.z) + 5),
                    [self.x >= [2, 3], self.z <= [-1, -4]])
        result = p.solve(solver=cp.ECOS)
        self.assertAlmostEqual(float(result), 12)
        self.assertAlmostEqual(float(list(self.x.value)[1] - list(self.z.value)[1]), 7)

    # Test problems with norm1
    def test_norm1(self) -> None:
        # Constant argument.
        p = Problem(cp.Minimize(cp.norm1(-2)))
        result = p.solve()
        self.assertAlmostEqual(result, 2)

        # Scalar arguments.
        p = Problem(cp.Minimize(cp.norm1(self.a)), [self.a <= -2])
        result = p.solve()
        self.assertAlmostEqual(result, 2)
        self.assertAlmostEqual(self.a.value, -2)

        # cp.Maximize
        p = Problem(cp.Maximize(-cp.norm1(self.a)), [self.a <= -2])
        result = p.solve()
        self.assertAlmostEqual(result, -2)
        self.assertAlmostEqual(self.a.value, -2)

        # Vector arguments.
        p = Problem(cp.Minimize(cp.norm1(self.x - self.z) + 5),
                    [self.x >= [2, 3], self.z <= [-1, -4]])
        result = p.solve()
        self.assertAlmostEqual(float(result), 15)
        self.assertAlmostEqual(float(list(self.x.value)[1] - list(self.z.value)[1]), 7)

    # Test problems with norm2
    def test_norm2(self) -> None:
        # Constant argument.
        p = Problem(cp.Minimize(cp.pnorm(-2, p=2)))
        result = p.solve()
        self.assertAlmostEqual(result, 2)

        # Scalar arguments.
        p = Problem(cp.Minimize(cp.pnorm(self.a, p=2)), [self.a <= -2])
        result = p.solve()
        self.assertAlmostEqual(result, 2)
        self.assertAlmostEqual(self.a.value, -2)

        # cp.Maximize
        p = Problem(cp.Maximize(-cp.pnorm(self.a, p=2)), [self.a <= -2])
        result = p.solve()
        self.assertAlmostEqual(result, -2)
        self.assertAlmostEqual(self.a.value, -2)

        # Vector arguments.
        p = Problem(cp.Minimize(cp.pnorm(self.x - self.z, p=2) + 5),
                    [self.x >= [2, 3], self.z <= [-1, -4]])
        result = p.solve()
        self.assertAlmostEqual(result, 12.61577)
        self.assertItemsAlmostEqual(self.x.value, [2, 3])
        self.assertItemsAlmostEqual(self.z.value, [-1, -4])

        # Row  arguments.
        p = Problem(cp.Minimize(cp.pnorm((self.x - self.z).T, p=2) + 5),
                    [self.x >= [2, 3], self.z <= [-1, -4]])
        result = p.solve()
        self.assertAlmostEqual(result, 12.61577)
        self.assertItemsAlmostEqual(self.x.value, [2, 3])
        self.assertItemsAlmostEqual(self.z.value, [-1, -4])

    # Test problems with abs
    def test_abs(self) -> None:
        p = Problem(cp.Minimize(cp.sum(cp.abs(self.A))), [-2 >= self.A])
        result = p.solve()
        self.assertAlmostEqual(result, 8)
        self.assertItemsAlmostEqual(self.A.value, [-2, -2, -2, -2])

    # Test problems with quad_form.
    def test_quad_form(self) -> None:
        with self.assertRaises(Exception) as cm:
            Problem(cp.Minimize(cp.quad_form(self.x, self.A))).solve()
        self.assertEqual(
            str(cm.exception),
            "At least one argument to quad_form must be non-variable."
        )

        with self.assertRaises(Exception) as cm:
            Problem(cp.Minimize(cp.quad_form(1, self.A))).solve()
        self.assertEqual(str(cm.exception), "Invalid dimensions for arguments.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises(Exception) as cm:
                Problem(cp.Minimize(cp.quad_form(self.x, [[-1, 0], [0, 9]]))).solve()
            self.assertTrue("Problem does not follow DCP rules."
                            in str(cm.exception))

        P = [[4, 0], [0, 9]]
        p = Problem(cp.Minimize(cp.quad_form(self.x, P)), [self.x >= 1])
        result = p.solve()
        self.assertAlmostEqual(result, 13, places=3)

        c = [1, 2]
        p = Problem(cp.Minimize(cp.quad_form(c, self.A)), [self.A >= 1])
        result = p.solve()
        self.assertAlmostEqual(result, 9)

        c = [1, 2]
        P = [[4, 0], [0, 9]]
        p = Problem(cp.Minimize(cp.quad_form(c, P)))
        result = p.solve()
        self.assertAlmostEqual(result, 40)

    # Test combining atoms
    def test_mixed_atoms(self) -> None:
        p = Problem(cp.Minimize(cp.pnorm(5 + cp.norm1(self.z)
                                         + cp.norm1(self.x) +
                                         cp.norm_inf(self.x - self.z), p=2)),
                    [self.x >= [2, 3], self.z <= [-1, -4], cp.pnorm(self.x + self.z, p=2) <= 2])
        result = p.solve()
        self.assertAlmostEqual(result, 22)
        self.assertItemsAlmostEqual(self.x.value, [2, 3])
        self.assertItemsAlmostEqual(self.z.value, [-1, -4])

    # Test multiplying by constant atoms.
    def test_mult_constant_atoms(self) -> None:
        p = Problem(cp.Minimize(cp.pnorm([3, 4], p=2)*self.a), [self.a >= 2])
        result = p.solve()
        self.assertAlmostEqual(result, 10)
        self.assertAlmostEqual(self.a.value, 2)

    def test_dual_variables(self) -> None:
        """Test recovery of dual variables.
        """
        for solver in [s.ECOS, s.SCS, s.CVXOPT]:
            if solver in INSTALLED_SOLVERS:
                if solver == s.SCS:
                    acc = 1
                else:
                    acc = 5
                p = Problem(cp.Minimize(cp.norm1(self.x + self.z)),
                            [self.x >= [2, 3],
                            [[1, 2], [3, 4]] @ self.z == [-1, -4],
                            cp.pnorm(self.x + self.z, p=2) <= 100])
                result = p.solve(solver=solver)
                self.assertAlmostEqual(result, 4, places=acc)
                self.assertItemsAlmostEqual(self.x.value, [4, 3], places=acc)
                self.assertItemsAlmostEqual(self.z.value, [-4, 1], places=acc)
                # Dual values
                self.assertItemsAlmostEqual(p.constraints[0].dual_value, [0, 1], places=acc)
                self.assertItemsAlmostEqual(p.constraints[1].dual_value, [-1, 0.5], places=acc)
                self.assertAlmostEqual(p.constraints[2].dual_value, 0, places=acc)

                T = numpy.ones((2, 3))*2
                p = Problem(cp.Minimize(1),
                            [self.A >= T @ self.C,
                             self.A == self.B,
                             self.C == T.T])
                result = p.solve(solver=solver)
                # Dual values
                self.assertItemsAlmostEqual(p.constraints[0].dual_value, 4*[0], places=acc)
                self.assertItemsAlmostEqual(p.constraints[1].dual_value, 4*[0], places=acc)
                self.assertItemsAlmostEqual(p.constraints[2].dual_value, 6*[0], places=acc)

    # Test problems with indexing.
    def test_indexing(self) -> None:
        # Vector variables
        p = Problem(cp.Maximize(self.x[0]), [self.x[0] <= 2, self.x[1] == 3])
        result = p.solve()
        self.assertAlmostEqual(result, 2)
        self.assertItemsAlmostEqual(self.x.value, [2, 3])

        n = 10
        A = numpy.arange(n*n)
        A = numpy.reshape(A, (n, n))
        x = Variable((n, n))
        p = Problem(cp.Minimize(cp.sum(x)), [x == A])
        result = p.solve()
        answer = n*n*(n*n+1)/2 - n*n
        self.assertAlmostEqual(result, answer)

        # Matrix variables
        p = Problem(cp.Maximize(sum(self.A[i, i] + self.A[i, 1-i] for i in range(2))),
                    [self.A <= [[1, -2], [-3, 4]]])
        result = p.solve()
        self.assertAlmostEqual(result, 0)
        self.assertItemsAlmostEqual(self.A.value, [1, -2, -3, 4])

        # Indexing arithmetic expressions.
        expr = [[1, 2], [3, 4]] @ self.z + self.x
        p = Problem(cp.Minimize(expr[1]), [self.x == self.z, self.z == [1, 2]])
        result = p.solve()
        self.assertAlmostEqual(result, 12)
        self.assertItemsAlmostEqual(self.x.value, self.z.value)

    def test_non_python_int_index(self) -> None:
        """Test problems that have special types as indices.
        """
        import sys
        if sys.version_info > (3,):
            my_long = int
        else:
            my_long = long  # noqa: F821
        # Test with long indices.
        cost = self.x[0:my_long(2)][0]
        p = Problem(cp.Minimize(cost), [self.x == 1])
        result = p.solve()
        self.assertAlmostEqual(result, 1)
        self.assertItemsAlmostEqual(self.x.value, [1, 1])

        # Test with numpy64 indices.
        cost = self.x[0:numpy.int64(2)][0]
        p = Problem(cp.Minimize(cost), [self.x == 1])
        result = p.solve()
        self.assertAlmostEqual(result, 1)
        self.assertItemsAlmostEqual(self.x.value, [1, 1])

    # Test problems with slicing.
    def test_slicing(self) -> None:
        p = Problem(cp.Maximize(cp.sum(self.C)), [self.C[1:3, :] <= 2, self.C[0, :] == 1])
        result = p.solve()
        self.assertAlmostEqual(result, 10)
        self.assertItemsAlmostEqual(self.C.value, 2*[1, 2, 2])

        p = Problem(cp.Maximize(cp.sum(self.C[0:3:2, 1])),
                    [self.C[1:3, :] <= 2, self.C[0, :] == 1])
        result = p.solve()
        self.assertAlmostEqual(result, 3)
        self.assertItemsAlmostEqual(self.C.value[0:3:2, 1], [1, 2])

        p = Problem(cp.Maximize(cp.sum((self.C[0:2, :] + self.A)[:, 0:2])),
                    [self.C[1:3, :] <= 2, self.C[0, :] == 1,
                     (self.A + self.B)[:, 0] == 3, (self.A + self.B)[:, 1] == 2,
                     self.B == 1])
        result = p.solve()
        self.assertAlmostEqual(result, 12)
        self.assertItemsAlmostEqual(self.C.value[0:2, :], [1, 2, 1, 2])
        self.assertItemsAlmostEqual(self.A.value, [2, 2, 1, 1])

        p = Problem(cp.Maximize([[3], [4]] @ (self.C[0:2, :] + self.A)[:, 0]),
                    [self.C[1:3, :] <= 2, self.C[0, :] == 1,
                     [[1], [2]] @ (self.A + self.B)[:, 0] == 3, (self.A + self.B)[:, 1] == 2,
                     self.B == 1, 3*self.A[:, 0] <= 3])
        result = p.solve()
        self.assertAlmostEqual(result, 12)
        self.assertItemsAlmostEqual(self.C.value[0:2, 0], [1, 2])
        self.assertItemsAlmostEqual(self.A.value, [1, -.5, 1, 1])

        p = Problem(cp.Minimize(cp.pnorm((self.C[0:2, :] + self.A)[:, 0], p=2)),
                    [self.C[1:3, :] <= 2, self.C[0, :] == 1,
                     (self.A + self.B)[:, 0] == 3, (self.A + self.B)[:, 1] == 2,
                     self.B == 1])
        result = p.solve()
        self.assertAlmostEqual(result, 3)
        self.assertItemsAlmostEqual(self.C.value[0:2, 0], [1, -2], places=3)
        self.assertItemsAlmostEqual(self.A.value, [2, 2, 1, 1])

        # Transpose of slice.
        p = Problem(cp.Maximize(cp.sum(self.C)), [self.C[1:3, :].T <= 2, self.C[0, :].T == 1])
        result = p.solve()
        self.assertAlmostEqual(result, 10)
        self.assertItemsAlmostEqual(self.C.value, 2*[1, 2, 2])

    # Test the vstack atom.
    def test_vstack(self) -> None:
        a = Variable((1, 1), name='a')
        b = Variable((1, 1), name='b')

        x = Variable((2, 1), name='x')
        y = Variable((3, 1), name='y')

        c = numpy.ones((1, 5))
        p = Problem(cp.Minimize(c @ cp.vstack([x, y])),
                    [x == [[1, 2]],
                     y == [[3, 4, 5]]])
        result = p.solve()
        self.assertAlmostEqual(result, 15)

        c = numpy.ones((1, 4))
        p = Problem(cp.Minimize(c @ cp.vstack([x, x])),
                    [x == [[1, 2]]])
        result = p.solve()
        self.assertAlmostEqual(result, 6)

        c = numpy.ones((2, 2))
        p = Problem(cp.Minimize(cp.sum(cp.vstack([self.A, self.C]))),
                    [self.A >= 2*c,
                     self.C == -2])
        result = p.solve()
        self.assertAlmostEqual(result, -4)

        c = numpy.ones((1, 2))
        p = Problem(cp.Minimize(cp.sum(cp.vstack([c @ self.A, c @ self.B]))),
                    [self.A >= 2,
                     self.B == -2])
        result = p.solve()
        self.assertAlmostEqual(result, 0)

        c = numpy.array([[1, -1]]).T
        p = Problem(cp.Minimize(c.T @ cp.vstack([cp.square(a), cp.sqrt(b)])),
                    [a == 2,
                     b == 16])
        with self.assertRaises(Exception) as cm:
            p.solve()
        self.assertTrue("Problem does not follow DCP rules."
                        in str(cm.exception))

    # Test the hstack atom.
    def test_hstack(self) -> None:
        a = Variable((1, 1), name='a')
        b = Variable((1, 1), name='b')

        x = Variable((2, 1), name='x')
        y = Variable((3, 1), name='y')

        c = numpy.ones((1, 5))
        p = Problem(cp.Minimize(c @ cp.hstack([x.T, y.T]).T),
                    [x == [[1, 2]],
                     y == [[3, 4, 5]]])
        result = p.solve()
        self.assertAlmostEqual(result, 15)

        c = numpy.ones((1, 4))
        p = Problem(cp.Minimize(c @ cp.hstack([x.T, x.T]).T),
                    [x == [[1, 2]]])
        result = p.solve()
        self.assertAlmostEqual(result, 6)

        c = numpy.ones((2, 2))
        p = Problem(cp.Minimize(cp.sum(cp.hstack([self.A.T, self.C.T]))),
                    [self.A >= 2*c,
                     self.C == -2])
        result = p.solve()
        self.assertAlmostEqual(result, -4)

        D = Variable((3, 3))
        expr = cp.hstack([self.C, D])
        p = Problem(cp.Minimize(expr[0, 1] + cp.sum(cp.hstack([expr, expr]))),
                    [self.C >= 0,
                     D >= 0, D[0, 0] == 2, self.C[0, 1] == 3])
        result = p.solve()
        self.assertAlmostEqual(result, 13)

        c = numpy.array([[1, -1]]).T
        p = Problem(cp.Minimize(c.T @ cp.hstack([cp.square(a).T, cp.sqrt(b).T]).T),
                    [a == 2,
                     b == 16])
        with self.assertRaises(Exception) as cm:
            p.solve()
        self.assertTrue("Problem does not follow DCP rules."
                        in str(cm.exception))

    def test_bad_objective(self) -> None:
        """Test using a cvxpy expression as an objective.
        """
        with self.assertRaises(Exception) as cm:
            Problem(self.x+2)
        self.assertEqual(str(cm.exception), "Problem objective must be Minimize or Maximize.")

    # Test variable transpose.
    def test_transpose(self) -> None:
        p = Problem(cp.Minimize(cp.sum(self.x)),
                    [self.x[None, :] >= numpy.array([[1, 2]])])
        result = p.solve()
        self.assertAlmostEqual(result, 3)
        self.assertItemsAlmostEqual(self.x.value, [1, 2])

        p = Problem(cp.Minimize(cp.sum(self.C)),
                    [numpy.array([[1, 1]]) @ self.C.T >= numpy.array([[0, 1, 2]])])
        result = p.solve()
        value = self.C.value

        constraints = [1*self.C[i, 0] + 1*self.C[i, 1] >= i for i in range(3)]
        p = Problem(cp.Minimize(cp.sum(self.C)), constraints)
        result2 = p.solve()
        self.assertAlmostEqual(result, result2)
        self.assertItemsAlmostEqual(self.C.value, value)

        p = Problem(cp.Minimize(self.A[0, 1] - self.A.T[1, 0]),
                    [self.A == [[1, 2], [3, 4]]])
        result = p.solve()
        self.assertAlmostEqual(result, 0)

        p = Problem(cp.Minimize(cp.sum(self.x)), [(-self.x).T <= 1])
        result = p.solve()
        self.assertAlmostEqual(result, -2)

        c = numpy.array([[1, -1]]).T
        p = Problem(cp.Minimize(cp.maximum(c.T, 2, 2 + c.T)[0, 1]))
        result = p.solve()
        self.assertAlmostEqual(result, 2)

        c = numpy.array([[1, -1, 2], [1, -1, 2]]).T
        p = Problem(cp.Minimize(cp.sum(cp.maximum(c, 2, 2 + c).T[:, 0])))
        result = p.solve()
        self.assertAlmostEqual(result, 6)

        c = numpy.array([[1, -1, 2], [1, -1, 2]]).T
        p = Problem(cp.Minimize(cp.sum(cp.square(c.T).T[:, 0])))
        result = p.solve()
        self.assertAlmostEqual(result, 6)

        # Slice of transpose.
        p = Problem(cp.Maximize(cp.sum(self.C)), [self.C.T[:, 1:3] <= 2, self.C.T[:, 0] == 1])
        result = p.solve()
        self.assertAlmostEqual(result, 10)
        self.assertItemsAlmostEqual(self.C.value, 2*[1, 2, 2])

    def test_multiplication_on_left(self) -> None:
        """Test multiplication on the left by a non-constant.
        """
        c = numpy.array([[1, 2]]).T
        p = Problem(cp.Minimize(c.T @ self.A @ c), [self.A >= 2])
        result = p.solve()
        self.assertAlmostEqual(result, 18)

        p = Problem(cp.Minimize(self.a*2), [self.a >= 2])
        result = p.solve()
        self.assertAlmostEqual(result, 4)

        p = Problem(cp.Minimize(self.x.T @ c), [self.x >= 2])
        result = p.solve()
        self.assertAlmostEqual(result, 6)

        p = Problem(cp.Minimize((self.x.T + self.z.T) @ c),
                    [self.x >= 2, self.z >= 1])
        result = p.solve()
        self.assertAlmostEqual(result, 9)

        # TODO segfaults in Python 3
        A = numpy.ones((5, 10))
        x = Variable(5)
        p = cp.Problem(cp.Minimize(cp.sum(x @ A)), [x >= 0])
        result = p.solve()
        self.assertAlmostEqual(result, 0)

    # Test redundant constraints in cpopt.
    def test_redundant_constraints(self) -> None:
        obj = cp.Minimize(cp.sum(self.x))
        constraints = [self.x == 2, self.x == 2, self.x.T == 2, self.x[0] == 2]
        p = Problem(obj, constraints)
        result = p.solve(solver=s.SCS)
        self.assertAlmostEqual(result, 4)

        obj = cp.Minimize(cp.sum(cp.square(self.x)))
        constraints = [self.x == self.x]
        p = Problem(obj, constraints)
        result = p.solve(solver=s.SCS)
        self.assertAlmostEqual(result, 0)

        with self.assertRaises(ValueError) as cm:
            obj = cp.Minimize(cp.sum(cp.square(self.x)))
            constraints = [self.x == self.x]
            problem = Problem(obj, constraints)
            problem.solve(solver=s.ECOS)
        self.assertEqual(
            str(cm.exception),
            "ECOS cannot handle sparse data with nnz == 0; "
            "this is a bug in ECOS, and it indicates that your problem "
            "might have redundant constraints.")

    # Test that symmetry is enforced.
    def test_sdp_symmetry(self) -> None:
        p = Problem(cp.Minimize(cp.lambda_max(self.A)), [self.A >= 2])
        p.solve()
        self.assertItemsAlmostEqual(self.A.value, self.A.value.T, places=3)

        p = Problem(cp.Minimize(cp.lambda_max(self.A)), [self.A == [[1, 2], [3, 4]]])
        p.solve()
        self.assertEqual(p.status, s.INFEASIBLE)

    # Test PSD
    def test_sdp(self) -> None:
        # Ensure sdp constraints enforce transpose.
        obj = cp.Maximize(self.A[1, 0] - self.A[0, 1])
        p = Problem(obj, [cp.lambda_max(self.A) <= 100,
                          self.A[0, 0] == 2,
                          self.A[1, 1] == 2,
                          self.A[1, 0] == 2])
        result = p.solve()
        self.assertAlmostEqual(result, 0, places=3)

    # Test getting values for expressions.
    def test_expression_values(self) -> None:
        diff_exp = self.x - self.z
        inf_exp = cp.norm_inf(diff_exp)
        sum_exp = 5 + cp.norm1(self.z) + cp.norm1(self.x) + inf_exp
        constr_exp = cp.pnorm(self.x + self.z, p=2)
        obj = cp.pnorm(sum_exp, p=2)
        p = Problem(cp.Minimize(obj),
                    [self.x >= [2, 3], self.z <= [-1, -4], constr_exp <= 2])
        result = p.solve()
        self.assertAlmostEqual(result, 22)
        self.assertItemsAlmostEqual(self.x.value, [2, 3])
        self.assertItemsAlmostEqual(self.z.value, [-1, -4])
        # Expression values.
        self.assertItemsAlmostEqual(diff_exp.value, self.x.value - self.z.value)
        self.assertAlmostEqual(inf_exp.value,
                               LA.norm(self.x.value - self.z.value, numpy.inf))
        self.assertAlmostEqual(sum_exp.value,
                               5 + LA.norm(self.z.value, 1) + LA.norm(self.x.value, 1) +
                               LA.norm(self.x.value - self.z.value, numpy.inf))
        self.assertAlmostEqual(constr_exp.value,
                               LA.norm(self.x.value + self.z.value, 2))
        self.assertAlmostEqual(obj.value, result)

    def test_mult_by_zero(self) -> None:
        """Test multiplication by zero.
        """
        self.a.value = 1
        exp = 0*self.a
        self.assertEqual(exp.value, 0)
        obj = cp.Minimize(exp)
        p = Problem(obj)
        result = p.solve()
        self.assertAlmostEqual(result, 0)
        assert self.a.value is not None

    def test_div(self) -> None:
        """Tests a problem with division.
        """
        obj = cp.Minimize(cp.norm_inf(self.A/5))
        p = Problem(obj, [self.A >= 5])
        result = p.solve()
        self.assertAlmostEqual(result, 1)

        c = cp.Constant([[1., -1], [2, -2]])
        expr = self.A/(1./c)
        obj = cp.Minimize(cp.norm_inf(expr))
        p = Problem(obj, [self.A == 5])
        result = p.solve()
        self.assertAlmostEqual(result, 10)
        self.assertItemsAlmostEqual(expr.value, [5, -5] + [10, -10])

        # Test with a sparse matrix.
        import scipy.sparse as sp
        interface = intf.get_matrix_interface(sp.csc_matrix)
        c = interface.const_to_matrix([1, 2])
        c = cp.Constant(c)
        expr = self.x[:, None]/(1/c)
        obj = cp.Minimize(cp.norm_inf(expr))
        p = Problem(obj, [self.x == 5])
        result = p.solve()
        self.assertAlmostEqual(result, 10)
        self.assertItemsAlmostEqual(expr.value, [5, 10])

        # Test promotion.
        c = [[1, -1], [2, -2]]
        c = cp.Constant(c)
        expr = self.a/(1/c)
        obj = cp.Minimize(cp.norm_inf(expr))
        p = Problem(obj, [self.a == 5])
        result = p.solve()
        self.assertAlmostEqual(result, 10)
        self.assertItemsAlmostEqual(expr.value, [5, -5] + [10, -10])

    def test_multiply(self) -> None:
        """Tests problems with multiply.
        """
        c = [[1, -1], [2, -2]]
        expr = cp.multiply(c, self.A)
        obj = cp.Minimize(cp.norm_inf(expr))
        p = Problem(obj, [self.A == 5])
        result = p.solve()
        self.assertAlmostEqual(result, 10)
        self.assertItemsAlmostEqual(expr.value, [5, -5] + [10, -10])

        # Test with a sparse matrix.
        import scipy.sparse as sp
        interface = intf.get_matrix_interface(sp.csc_matrix)
        c = interface.const_to_matrix([1, 2])
        expr = cp.multiply(c, self.x[:, None])
        obj = cp.Minimize(cp.norm_inf(expr))
        p = Problem(obj, [self.x == 5])
        result = p.solve()
        self.assertAlmostEqual(result, 10)
        self.assertItemsAlmostEqual(expr.value.toarray(), [5, 10])

        # Test promotion.
        c = [[1, -1], [2, -2]]
        expr = cp.multiply(c, self.a)
        obj = cp.Minimize(cp.norm_inf(expr))
        p = Problem(obj, [self.a == 5])
        result = p.solve()
        self.assertAlmostEqual(result, 10)
        self.assertItemsAlmostEqual(expr.value, [5, -5] + [10, -10])

    def test_invalid_solvers(self) -> None:
        """Tests that errors occur when you use an invalid solver.
        """
        with self.assertRaises(SolverError):
            Problem(cp.Minimize(Variable(boolean=True))).solve(solver=s.ECOS)

        with self.assertRaises(SolverError):
            Problem(cp.Minimize(cp.lambda_max(self.A))).solve(solver=s.ECOS)

        with self.assertRaises(SolverError):
            Problem(cp.Minimize(self.a)).solve(solver=s.SCS)

    def test_solver_error_raised_on_failure(self) -> None:
        """Tests that a SolverError is raised when a solver fails.
        """
        A = numpy.random.randn(40, 40)
        b = cp.matmul(A, numpy.random.randn(40))

        with self.assertRaises(SolverError):
            Problem(cp.Minimize(
                cp.sum_squares(cp.matmul(A, cp.Variable(40)) - b))).solve(
                solver=s.OSQP, max_iter=1)

    def test_reshape(self) -> None:
        """Tests problems with reshape.
        """
        # Test on scalars.
        self.assertEqual(cp.reshape(1, (1, 1)).value, 1)

        # Test vector to matrix.
        x = Variable(4)
        mat = numpy.array([[1, -1], [2, -2]]).T
        vec = numpy.array([[1, 2, 3, 4]]).T
        vec_mat = numpy.array([[1, 2], [3, 4]]).T
        expr = cp.reshape(x, (2, 2))
        obj = cp.Minimize(cp.sum(mat @ expr))
        prob = Problem(obj, [x[:, None] == vec])
        result = prob.solve()
        self.assertAlmostEqual(result, numpy.sum(mat.dot(vec_mat)))

        # Test on matrix to vector.
        c = [1, 2, 3, 4]
        expr = cp.reshape(self.A, (4, 1))
        obj = cp.Minimize(expr.T @ c)
        constraints = [self.A == [[-1, -2], [3, 4]]]
        prob = Problem(obj, constraints)
        result = prob.solve()
        self.assertAlmostEqual(result, 20)
        self.assertItemsAlmostEqual(expr.value, [-1, -2, 3, 4])
        self.assertItemsAlmostEqual(cp.reshape(expr, (2, 2)).value, [-1, -2, 3, 4])

        # Test matrix to matrix.
        expr = cp.reshape(self.C, (2, 3))
        mat = numpy.array([[1, -1], [2, -2]])
        C_mat = numpy.array([[1, 4], [2, 5], [3, 6]])
        obj = cp.Minimize(cp.sum(mat @ expr))
        prob = Problem(obj, [self.C == C_mat])
        result = prob.solve()
        reshaped = numpy.reshape(C_mat, (2, 3), 'F')
        self.assertAlmostEqual(result, (mat.dot(reshaped)).sum())
        self.assertItemsAlmostEqual(expr.value, C_mat)

        # Test promoted expressions.
        c = numpy.array([[1, -1], [2, -2]]).T
        expr = cp.reshape(c * self.a, (1, 4))
        obj = cp.Minimize(expr @ [1, 2, 3, 4])
        prob = Problem(obj, [self.a == 2])
        result = prob.solve()
        self.assertAlmostEqual(result, -6)
        self.assertItemsAlmostEqual(expr.value, 2*c)

        expr = cp.reshape(c * self.a, (4, 1))
        obj = cp.Minimize(expr.T @ [1, 2, 3, 4])
        prob = Problem(obj, [self.a == 2])
        result = prob.solve()
        self.assertAlmostEqual(result, -6)
        self.assertItemsAlmostEqual(expr.value, 2*c)

    def test_cumsum(self) -> None:
        """Test problems with cumsum.
        """
        tt = cp.Variable(5)
        prob = cp.Problem(cp.Minimize(cp.sum(tt)),
                          [cp.cumsum(tt, 0) >= -0.0001])
        result = prob.solve()
        self.assertAlmostEqual(result, -0.0001)

    def test_cummax(self) -> None:
        """Test problems with cummax.
        """
        tt = cp.Variable(5)
        prob = cp.Problem(cp.Maximize(cp.sum(tt)),
                          [cp.cummax(tt, 0) <= numpy.array([1, 2, 3, 4, 5])])
        result = prob.solve()
        self.assertAlmostEqual(result, 15)

    def test_vec(self) -> None:
        """Tests problems with vec.
        """
        c = [1, 2, 3, 4]
        expr = cp.vec(self.A)
        obj = cp.Minimize(expr.T @ c)
        constraints = [self.A == [[-1, -2], [3, 4]]]
        prob = Problem(obj, constraints)
        result = prob.solve()
        self.assertAlmostEqual(result, 20)
        self.assertItemsAlmostEqual(expr.value, [-1, -2, 3, 4])

    def test_diag_prob(self) -> None:
        """Test a problem with diag.
        """
        C = Variable((3, 3))
        obj = cp.Maximize(C[0, 2])
        constraints = [cp.diag(C) == 1,
                       C[0, 1] == 0.6,
                       C[1, 2] == -0.3,
                       C == Variable((3, 3), PSD=True)]
        prob = Problem(obj, constraints)
        result = prob.solve()
        self.assertAlmostEqual(result, 0.583151, places=2)

    def test_presolve_parameters(self) -> None:
        """Test presolve with parameters.
        """
        # Test with parameters.
        gamma = Parameter(nonneg=True)
        x = Variable()
        obj = cp.Minimize(x)
        prob = Problem(obj, [gamma == 1, x >= 0])
        gamma.value = 0
        prob.solve(solver=s.SCS)
        self.assertEqual(prob.status, s.INFEASIBLE)

        gamma.value = 1
        prob.solve(solver=s.SCS)
        self.assertEqual(prob.status, s.OPTIMAL)

    def test_parameter_expressions(self) -> None:
        """Test that expressions with parameters are updated properly.
        """
        x = Variable()
        y = Variable()
        x0 = Parameter()
        xSquared = x0*x0 + 2*x0*(x - x0)

        # initial guess for x
        x0.value = 2

        # make the constraint x**2 - y == 0
        g = xSquared - y

        # set up the problem
        obj = cp.abs(x - 1)
        prob = Problem(cp.Minimize(obj), [g == 0])
        self.assertFalse(prob.is_dpp())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob.solve(cp.SCS)
        x0.value = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob.solve()
        self.assertAlmostEqual(g.value, 0)

        # Test multiplication.
        prob = Problem(cp.Minimize(x0*x), [x == 1])
        x0.value = 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob.solve()
        x0.value = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob.solve()
        self.assertAlmostEqual(prob.value, 1, places=2)

    def test_psd_constraints(self) -> None:
        """Test positive definite constraints.
        """
        C = Variable((3, 3))
        obj = cp.Maximize(C[0, 2])
        constraints = [cp.diag(C) == 1,
                       C[0, 1] == 0.6,
                       C[1, 2] == -0.3,
                       C == C.T,
                       C >> 0]
        prob = Problem(obj, constraints)
        result = prob.solve()
        self.assertAlmostEqual(result, 0.583151, places=2)

        C = Variable((2, 2))
        obj = cp.Maximize(C[0, 1])
        constraints = [C == 1, C >> [[2, 0], [0, 2]]]
        prob = Problem(obj, constraints)
        result = prob.solve()
        self.assertEqual(prob.status, s.INFEASIBLE)

        C = Variable((2, 2), symmetric=True)
        obj = cp.Minimize(C[0, 0])
        constraints = [C << [[2, 0], [0, 2]]]
        prob = Problem(obj, constraints)
        result = prob.solve()
        self.assertEqual(prob.status, s.UNBOUNDED)

    def test_psd_duals(self) -> None:
        """Test the duals of PSD constraints.
        """
        if s.CVXOPT in INSTALLED_SOLVERS:
            # Test the dual values with cpopt.
            C = Variable((2, 2), symmetric=True, name='C')
            obj = cp.Maximize(C[0, 0])
            constraints = [C << [[2, 0], [0, 2]]]
            prob = Problem(obj, constraints)
            result = prob.solve(solver=s.CVXOPT)
            self.assertAlmostEqual(result, 2)

            psd_constr_dual = constraints[0].dual_value.copy()
            C = Variable((2, 2), symmetric=True, name='C')
            X = Variable((2, 2), PSD=True)
            obj = cp.Maximize(C[0, 0])
            constraints = [X == [[2, 0], [0, 2]] - C]
            prob = Problem(obj, constraints)
            result = prob.solve(solver=s.CVXOPT)
            self.assertItemsAlmostEqual(constraints[0].dual_value, psd_constr_dual)

        # Test the dual values with SCS.
        C = Variable((2, 2), symmetric=True)
        obj = cp.Maximize(C[0, 0])
        constraints = [C << [[2, 0], [0, 2]]]
        prob = Problem(obj, constraints)
        result = prob.solve(solver=s.SCS)
        self.assertAlmostEqual(result, 2, places=4)

        psd_constr_dual = constraints[0].dual_value
        C = Variable((2, 2), symmetric=True)
        X = Variable((2, 2), PSD=True)
        obj = cp.Maximize(C[0, 0])
        constraints = [X == [[2, 0], [0, 2]] - C]
        prob = Problem(obj, constraints)
        result = prob.solve(solver=s.SCS)
        self.assertItemsAlmostEqual(constraints[0].dual_value, psd_constr_dual)

        # Test dual values with SCS that have off-diagonal entries.
        C = Variable((2, 2), symmetric=True)
        obj = cp.Maximize(C[0, 1] + C[1, 0])
        constraints = [C << [[2, 0], [0, 2]], C >= 0]
        prob = Problem(obj, constraints)
        result = prob.solve(solver=s.SCS)
        self.assertAlmostEqual(result, 4, places=3)

        psd_constr_dual = constraints[0].dual_value
        C = Variable((2, 2), symmetric=True)
        X = Variable((2, 2), PSD=True)
        obj = cp.Maximize(C[0, 1] + C[1, 0])
        constraints = [X == [[2, 0], [0, 2]] - C, C >= 0]
        prob = Problem(obj, constraints)
        result = prob.solve(solver=s.SCS)
        self.assertItemsAlmostEqual(constraints[0].dual_value, psd_constr_dual,
                                    places=3)

    def test_geo_mean(self) -> None:
        import numpy as np

        x = Variable(2)
        cost = cp.geo_mean(x)
        prob = Problem(cp.Maximize(cost), [x <= 1])
        prob.solve()
        self.assertAlmostEqual(prob.value, 1)

        prob = Problem(cp.Maximize(cost), [cp.sum(x) <= 1])
        prob.solve()
        self.assertItemsAlmostEqual(x.value, [.5, .5])

        x = Variable((3, 3))
        self.assertRaises(ValueError, cp.geo_mean, x)

        x = Variable((3, 1))
        g = cp.geo_mean(x)
        self.assertSequenceEqual(g.w, [Fraction(1, 3)]*3)

        x = Variable((1, 5))
        g = cp.geo_mean(x)
        self.assertSequenceEqual(g.w, [Fraction(1, 5)]*5)

        # check that we get the right answer for
        # max geo_mean(x) s.t. sum(x) <= 1
        p = np.array([.07, .12, .23, .19, .39])

        def short_geo_mean(x, p):
            p = np.array(p)/sum(p)
            x = np.array(x)
            return np.prod(x**p)

        x = Variable(5)
        prob = Problem(cp.Maximize(cp.geo_mean(x, p)), [cp.sum(x) <= 1])
        prob.solve()
        x = np.array(x.value).flatten()
        x_true = p/sum(p)

        self.assertTrue(np.allclose(prob.value, cp.geo_mean(list(x), p).value))
        self.assertTrue(np.allclose(prob.value, short_geo_mean(x, p)))
        self.assertTrue(np.allclose(x, x_true, 1e-3))

        # check that we get the right answer for
        # max geo_mean(x) s.t. norm(x) <= 1
        x = Variable(5)
        prob = Problem(cp.Maximize(cp.geo_mean(x, p)), [cp.norm(x) <= 1])
        prob.solve()
        x = np.array(x.value).flatten()
        x_true = np.sqrt(p/sum(p))

        self.assertTrue(np.allclose(prob.value, cp.geo_mean(list(x), p).value))
        self.assertTrue(np.allclose(prob.value, short_geo_mean(x, p)))
        self.assertTrue(np.allclose(x, x_true, 1e-3))

        # the following 3 tests check vstack and hstack input to geo_mean
        # the following 3 formulations should be equivalent
        n = 5
        x_true = np.ones(n)
        x = Variable(n)

        Problem(cp.Maximize(cp.geo_mean(x)), [x <= 1]).solve()
        xval = np.array(x.value).flatten()
        self.assertTrue(np.allclose(xval, x_true, 1e-3))

        y = cp.vstack([x[i] for i in range(n)])
        Problem(cp.Maximize(cp.geo_mean(y)), [x <= 1]).solve()
        xval = np.array(x.value).flatten()
        self.assertTrue(np.allclose(xval, x_true, 1e-3))

        y = cp.hstack([x[i] for i in range(n)])
        Problem(cp.Maximize(cp.geo_mean(y)), [x <= 1]).solve()
        xval = np.array(x.value).flatten()
        self.assertTrue(np.allclose(xval, x_true, 1e-3))

    def test_pnorm(self) -> None:
        import numpy as np

        x = Variable(3, name='x')

        a = np.array([1.0, 2, 3])

        # todo: add -1, .5, .3, -2.3 and testing positivity constraints

        for p in (1, 1.6, 1.3, 2, 1.99, 3, 3.7, np.inf):
            prob = Problem(cp.Minimize(cp.pnorm(x, p=p)), [x.T @ a >= 1])
            prob.solve(verbose=True)

            # formula is true for any a >= 0 with p > 1
            if p == np.inf:
                x_true = np.ones_like(a)/sum(a)
            elif p == 1:
                # only works for the particular a = [1,2,3]
                x_true = np.array([0, 0, 1.0/3])
            else:
                x_true = a**(1.0/(p-1))/a.dot(a**(1.0/(p-1)))

            x_alg = np.array(x.value).flatten()
            self.assertTrue(np.allclose(x_alg, x_true, 1e-2), 'p = {}'.format(p))
            self.assertTrue(np.allclose(prob.value, np.linalg.norm(x_alg, p)))
            self.assertTrue(np.allclose(np.linalg.norm(x_alg, p), cp.pnorm(x_alg, p).value))

    def test_pnorm_concave(self) -> None:
        import numpy as np

        x = Variable(3, name='x')

        # test positivity constraints
        a = np.array([-1.0, 2, 3])
        for p in (-1, .5, .3, -2.3):
            prob = Problem(cp.Minimize(cp.sum(cp.abs(x-a))), [cp.pnorm(x, p) >= 0])
            prob.solve()

            self.assertTrue(np.allclose(prob.value, 1))

        a = np.array([1.0, 2, 3])
        for p in (-1, .5, .3, -2.3):
            prob = Problem(cp.Minimize(cp.sum(cp.abs(x-a))), [cp.pnorm(x, p) >= 0])
            prob.solve()

            self.assertAlmostEqual(prob.value, 0, places=6)

    def test_power(self) -> None:
        x = Variable()
        prob = Problem(cp.Minimize(cp.power(x, 1.7) + cp.power(x, -2.3) - cp.power(x, .45)))
        prob.solve()
        x = x.value
        self.assertTrue(builtins.abs(1.7*x**.7 - 2.3*x**-3.3 - .45*x**-.55) <= 1e-3)

    def test_multiply_by_scalar(self) -> None:
        """Test a problem with multiply by a scalar.
        """
        import numpy as np
        T = 10
        J = 20
        rvec = np.random.randn(T, J)
        dy = np.random.randn(2*T)
        theta = Variable(J)

        delta = 1e-3
        loglambda = rvec @ theta  # rvec: TxJ regressor matrix, theta: (Jx1) cp variable
        a = cp.multiply(dy[0:T], loglambda)  # size(Tx1)
        b1 = cp.exp(loglambda)
        b2 = cp.multiply(delta, b1)
        cost = -a + b1

        cost = -a + b2  # size (Tx1)
        prob = Problem(cp.Minimize(cp.sum(cost)))
        prob.solve(solver=s.SCS)

        obj = cp.Minimize(cp.sum(cp.multiply(2, self.x)))
        prob = Problem(obj, [self.x == 2])
        result = prob.solve()
        self.assertAlmostEqual(result, 8)

    def test_int64(self) -> None:
        """Test bug with 64 bit integers.
        """
        q = cp.Variable(numpy.int64(2))
        objective = cp.Minimize(cp.norm(q, 1))
        problem = cp.Problem(objective)
        problem.solve()
        print(q.value)

    def test_neg_slice(self) -> None:
        """Test bug with negative slice.
        """
        x = cp.Variable(2)
        objective = cp.Minimize(x[0] + x[1])
        constraints = [x[-2:] >= 1]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        self.assertItemsAlmostEqual(x.value, [1, 1])

    def test_pnorm_axis(self) -> None:
        """Test pnorm with axis != 0.
        """
        b = numpy.arange(2)
        X = cp.Variable(shape=(2, 10))
        expr = cp.pnorm(X, p=2, axis=1) - b
        con = [expr <= 0]
        obj = cp.Maximize(cp.sum(X))
        prob = cp.Problem(obj, con)
        prob.solve(solver='ECOS')
        self.assertItemsAlmostEqual(expr.value, numpy.zeros(2))

        b = numpy.arange(10)
        X = cp.Variable(shape=(10, 2))
        expr = cp.pnorm(X, p=2, axis=1) - b
        con = [expr <= 0]
        obj = cp.Maximize(cp.sum(X))
        prob = cp.Problem(obj, con)
        prob.solve(solver='ECOS')
        self.assertItemsAlmostEqual(expr.value, numpy.zeros(10))

    def test_bool_constr(self) -> None:
        """Test constraints that evaluate to booleans.
        """
        x = cp.Variable(pos=True)
        prob = cp.Problem(cp.Minimize(x), [True])
        prob.solve()
        self.assertAlmostEqual(x.value, 0)

        x = cp.Variable(pos=True)
        prob = cp.Problem(cp.Minimize(x), [True]*10)
        prob.solve()
        self.assertAlmostEqual(x.value, 0)

        prob = cp.Problem(cp.Minimize(x), [42 <= x] + [True]*10)
        prob.solve()
        self.assertAlmostEqual(x.value, 42)

        prob = cp.Problem(cp.Minimize(x), [True] + [42 <= x] + [True] * 10)
        prob.solve()
        self.assertAlmostEqual(x.value, 42)

        prob = cp.Problem(cp.Minimize(x), [False])
        prob.solve()
        self.assertEqual(prob.status, s.INFEASIBLE)

        prob = cp.Problem(cp.Minimize(x), [False]*10)
        prob.solve()
        self.assertEqual(prob.status, s.INFEASIBLE)

        prob = cp.Problem(cp.Minimize(x), [True]*10 + [False] + [True]*10)
        prob.solve()
        self.assertEqual(prob.status, s.INFEASIBLE)

        prob = cp.Problem(cp.Minimize(x), [42 <= x] + [True]*10 + [False])
        prob.solve()
        self.assertEqual(prob.status, s.INFEASIBLE)

        # only Trues, but infeasible solution since x must be non-negative.
        prob = cp.Problem(cp.Minimize(x), [True] + [x <= -42] + [True]*10)
        prob.solve()
        self.assertEqual(prob.status, s.INFEASIBLE)

    def test_pos(self) -> None:
        """Test the pos and neg attributes.
        """
        x = cp.Variable(pos=True)
        prob = cp.Problem(cp.Minimize(x))
        prob.solve()
        self.assertAlmostEqual(x.value, 0)

        x = cp.Variable(neg=True)
        prob = cp.Problem(cp.Maximize(x))
        prob.solve()
        self.assertAlmostEqual(x.value, 0)

    def test_pickle(self) -> None:
        """Test pickling and unpickling problems.
        """
        prob = cp.Problem(cp.Minimize(2*self.a + 3),
                          [self.a >= 1])
        prob_str = pickle.dumps(prob)
        new_prob = pickle.loads(prob_str)
        result = new_prob.solve()
        self.assertAlmostEqual(result, 5.0)
        self.assertAlmostEqual(new_prob.variables()[0].value, 1.0)

    def test_spare_int8_matrix(self) -> None:
        """Test problem with sparse int8 matrix.
           issue #809.
        """

        a = Variable(shape=(3, 1))
        q = np.array([1.88922129, 0.06938685, 0.91948919])
        P = np.array([[280.64, -49.84, -80.],
                      [-49.84, 196.04, 139.],
                      [-80., 139., 106.]])
        D_dense = np.array([[-1, 1, 0, 0, 0, 0],
                            [0, -1, 1, 0, 0, 0],
                            [0, 0, 0, -1, 1, 0]], dtype=np.int8)
        D_sparse = sp.coo_matrix(D_dense)

        def make_problem(D):
            obj = cp.Minimize(0.5 * cp.quad_form(a, P) - a.T @ q)
            assert obj.is_dcp()

            alpha = cp.Parameter(nonneg=True, value=2)
            constraints = [a >= 0., -alpha <= D.T @ a, D.T @ a <= alpha]

            prob = cp.Problem(obj, constraints)
            prob.solve(solver=cp.settings.ECOS)
            assert prob.status == 'optimal'
            return prob

        expected_coef = np.array([
            [-0.011728003147, 0.011728002895, 0.000000000252,
             -0.017524801335, 0.017524801335, 0.]])

        make_problem(D_dense)
        coef_dense = a.value.T.dot(D_dense)
        np.testing.assert_almost_equal(expected_coef, coef_dense)

        make_problem(D_sparse)
        coef_sparse = a.value.T @ D_sparse
        np.testing.assert_almost_equal(expected_coef, coef_sparse)

    def test_special_index(self) -> None:
        """Test QP code path with special indexing.
        """
        x = cp.Variable((1, 3))
        y = cp.sum(x[:, 0:2], axis=1)
        cost = cp.QuadForm(y, np.diag([1]))
        prob = cp.Problem(cp.Minimize(cost))
        result1 = prob.solve()

        x = cp.Variable((1, 3))
        y = cp.sum(x[:, [0, 1]], axis=1)
        cost = cp.QuadForm(y, np.diag([1]))
        prob = cp.Problem(cp.Minimize(cost))
        result2 = prob.solve()
        self.assertAlmostEqual(result1, result2)

    def test_indicator(self) -> None:
        """Test a problem with indicators.
        """
        n = 5
        m = 2
        q = np.arange(n)
        a = np.ones((m, n))
        b = np.ones((m, 1))
        x = cp.Variable((n, 1), name='x')
        constraints = [a @ x == b]
        objective = cp.Minimize((1/2) * cp.square(q.T @ x) + cp.transforms.indicator(constraints))
        problem = cp.Problem(objective)
        solution1 = problem.solve()

        # Without indicators.
        objective = cp.Minimize((1/2) * cp.square(q.T @ x))
        problem = cp.Problem(objective, constraints)
        solution2 = problem.solve()
        self.assertAlmostEqual(solution1, solution2)

    def test_rmul_scalar_mats(self) -> None:
        """Test that rmul works with 1x1 matrices.
        """
        x = [[4144.30127531]]
        y = [[7202.52114311]]
        z = cp.Variable(shape=(1, 1))
        objective = cp.Minimize(cp.quad_form(z, x) - 2 * z.T @ y)

        prob = cp.Problem(objective)
        prob.solve('OSQP', verbose=True)
        result1 = prob.value

        x = 4144.30127531
        y = 7202.52114311
        z = cp.Variable()
        objective = cp.Minimize(x*z**2 - 2 * z * y)

        prob = cp.Problem(objective)
        prob.solve('OSQP', verbose=True)
        self.assertAlmostEqual(prob.value, result1)

    def test_min_with_axis(self) -> None:
        """Test reshape of a min with axis=0.
        """
        x = cp.Variable((5, 2))
        y = cp.Variable((5, 2))

        stacked_flattened = cp.vstack([cp.vec(x), cp.vec(y)])  # (2, 10)
        minimum = cp.min(stacked_flattened, axis=0)  # (10,)
        reshaped_minimum = cp.reshape(minimum, (5, 2))  # (5, 2)

        obj = cp.sum(reshaped_minimum)
        problem = cp.Problem(cp.Maximize(obj), [x == 1, y == 2])
        result = problem.solve()
        self.assertAlmostEqual(result, 10)

    def test_constant_infeasible(self) -> None:
        """Test a problem with constant values only that is infeasible.
        """
        p = cp.Problem(cp.Maximize(0), [cp.Constant(0) == 1])
        p.solve()
        self.assertEquals(p.status, cp.INFEASIBLE)

    def test_huber_scs(self) -> None:
        """Test that huber regression works with SCS.
           See issue #1370.
        """
        np.random.seed(1)
        m = 5
        n = 2

        x0 = np.random.randn(n)
        A = np.random.randn(m, n)
        b = A.dot(x0) + 0.01 * np.random.randn(m)
        # Add outlier noise.
        k = int(0.02 * m)
        idx = np.random.randint(m, size=k)
        b[idx] += 10 * np.random.randn(k)

        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.huber(A @ x - b))))

        prob.solve(solver=cp.SCS)
