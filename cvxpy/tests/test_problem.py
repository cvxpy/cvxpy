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
from cvxpy.atoms import *
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variables import Variable, Semidef, Bool, Symmetric
from cvxpy.problems.objective import *
from cvxpy.problems.problem import Problem
from cvxpy.problems.solvers.utilities import SOLVERS, installed_solvers
from cvxpy.problems.problem_data.sym_data import SymData
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.error import DCPError
from cvxpy.tests.base_test import BaseTest
from numpy import linalg as LA
import numpy
import sys
# Solvers.
import scs
import ecos
import warnings
PY2 = sys.version_info < (3, 0)
if sys.version_info < (3, 0):
    from cStringIO import StringIO
else:
    from io import StringIO

from nose.tools import set_trace


class TestProblem(BaseTest):
    """ Unit tests for the expression/expression module. """

    def setUp(self):
        self.a = Variable(name='a')
        self.b = Variable(name='b')
        self.c = Variable(name='c')

        self.x = Variable(2, name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(2, name='z')

        self.A = Variable(2, 2, name='A')
        self.B = Variable(2, 2, name='B')
        self.C = Variable(3, 2, name='C')

    def test_to_str(self):
        """Test string representations.
        """
        obj = Minimize(self.a)
        prob = Problem(obj)
        self.assertEqual(repr(prob), "Problem(%s, %s)" % (repr(obj), repr([])))
        constraints = [self.x*2 == self.x, self.x == 0]
        prob = Problem(obj, constraints)
        self.assertEqual(repr(prob), "Problem(%s, %s)" % (repr(obj), repr(constraints)))

        # Test str.
        result = "minimize %(name)s\nsubject to %(name)s == 0\n           0 <= %(name)s" % {"name": self.a.name()}
        prob = Problem(Minimize(self.a), [self.a == 0, self.a >= 0])
        self.assertEqual(str(prob), result)

    def test_variables(self):
        """Test the variables method.
        """
        p = Problem(Minimize(self.a), [self.a <= self.x, self.b <= self.A + 2])
        vars_ = p.variables()
        ref = [self.a, self.x, self.b, self.A]
        if PY2:
            self.assertItemsEqual(vars_, ref)
        else:
            self.assertCountEqual(vars_, ref)

    def test_parameters(self):
        """Test the parameters method.
        """
        p1 = Parameter()
        p2 = Parameter(3, sign="negative")
        p3 = Parameter(4, 4, sign="positive")
        p = Problem(Minimize(p1), [self.a + p1 <= p2, self.b <= p3 + p3 + 2])
        params = p.parameters()
        ref = [p1, p2, p3]
        if PY2:
            self.assertItemsEqual(params, ref)
        else:
            self.assertCountEqual(params, ref)

    def test_constants(self):
        """Test the constants method.
        """
        c1 = numpy.random.randn(1, 2)
        c2 = numpy.random.randn(2, 1)
        p = Problem(Minimize(c1*self.x), [self.x >= c2])
        constants_ = p.constants()
        ref = [c1, c2]
        self.assertEqual(len(ref), len(constants_))
        for c, r in zip(constants_, ref):
            self.assertTupleEqual(c.size, r.shape)
            self.assertTrue((c.value == r).all())
            # Allows comparison between numpy matrices and numpy arrays
            # Necessary because of the way cvxpy handles numpy arrays and constants

        # Single scalar constants
        p = Problem(Minimize(self.a), [self.x >= 1])
        constants_ = p.constants()
        ref = [numpy.matrix(1)]
        self.assertEqual(len(ref), len(constants_))
        for c, r in zip(constants_, ref):
            self.assertEqual(c.size, r.shape) and \
            self.assertTrue((c.value == r).all()) 
            # Allows comparison between numpy matrices and numpy arrays
            # Necessary because of the way cvxpy handles numpy arrays and constants


    def test_size_metrics(self):
        """Test the size_metrics method.
        """
        p1 = Parameter()
        p2 = Parameter(3, sign="negative")
        p3 = Parameter(4, 4, sign="positive")

        c1 = numpy.random.randn(2, 1)
        c2 = numpy.random.randn(1, 2)
        constants = [2, c2.dot(c1)]

        p = Problem(Minimize(p1), [self.a + p1 <= p2, self.b <= p3 + p3 + constants[0], self.c == constants[1]])
        # num_scalar_variables
        n_variables = p.size_metrics.num_scalar_variables
        ref = numpy.prod(self.a.size) + numpy.prod(self.b.size) + numpy.prod(self.c.size)
        if PY2:
            self.assertEqual(n_variables, ref)
        else:
            self.assertEqual(n_variables, ref)

        # num_scalar_data
        n_data = p.size_metrics.num_scalar_data
        ref = numpy.prod(p1.size) + numpy.prod(p2.size) + numpy.prod(p3.size) + len(constants)  # 2 and c2.dot(c1) are both single scalar constants.
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
        ref = max(p3.size)
        self.assertEqual(max_data_dim, ref)

    def test_solver_stats(self):
        """Test the solver_stats method.
        """
        prob = Problem(Minimize(norm(self.x)), [self.x == 0])
        prob.solve(solver = s.ECOS)
        stats = prob.solver_stats
        self.assertGreater(stats.solve_time, 0)
        self.assertGreater(stats.setup_time, 0)
        self.assertGreater(stats.num_iters, 0)

    def test_get_problem_data(self):
        """Test get_problem_data method.
        """
        with self.assertRaises(Exception) as cm:
            Problem(Maximize(Bool())).get_problem_data(s.ECOS)

        expected = (
            "The solver ECOS cannot solve the problem because"
            " it cannot solve mixed-integer problems."
        )
        self.assertEqual(str(cm.exception), expected)

        data = Problem(Maximize(exp(self.a) + 2)).get_problem_data(s.SCS)
        dims = data["dims"]
        self.assertEqual(dims['ep'], 1)
        self.assertEqual(data["c"].shape, (2,))
        self.assertEqual(data["A"].shape, (3, 2))

        data = Problem(Minimize(norm(self.x) + 3)).get_problem_data(s.ECOS)
        dims = data["dims"]
        self.assertEqual(dims["q"], [3])
        self.assertEqual(data["c"].shape, (3,))
        self.assertEqual(data["A"].shape, (0, 3))
        self.assertEqual(data["G"].shape, (3, 3))

        if s.CVXOPT in installed_solvers():
            import cvxopt
            data = Problem(Minimize(norm(self.x) + 3)).get_problem_data(s.CVXOPT)
            dims = data["dims"]
            self.assertEqual(dims["q"], [3])
            # NumPy ndarrays, not cvxopt matrices.
            self.assertEqual(type(data["c"]), cvxopt.matrix)
            self.assertEqual(type(data["A"]), cvxopt.spmatrix)
            self.assertEqual(data["c"].size, (3, 1))
            self.assertEqual(data["A"].size, (0, 3))
            self.assertEqual(data["G"].size, (3, 3))

    def test_unpack_results(self):
        """Test unpack results method.
        """
        with self.assertRaises(Exception) as cm:
            Problem(Minimize(exp(self.a))).unpack_results("blah", None)
        self.assertEqual(str(cm.exception), "Unknown solver.")

        prob = Problem(Minimize(exp(self.a)), [self.a == 0])
        args = prob.get_problem_data(s.SCS)
        data = {"c": args["c"], "A": args["A"], "b": args["b"]}
        results_dict = scs.solve(data, args["dims"])
        prob = Problem(Minimize(exp(self.a)), [self.a == 0])
        prob.unpack_results(s.SCS, results_dict)
        self.assertAlmostEqual(self.a.value, 0, places=3)
        self.assertAlmostEqual(prob.value, 1, places=3)
        self.assertAlmostEqual(prob.status, s.OPTIMAL)

        prob = Problem(Minimize(norm(self.x)), [self.x == 0])
        args = prob.get_problem_data(s.ECOS)
        results_dict = ecos.solve(args["c"], args["G"], args["h"],
                                  args["dims"], args["A"], args["b"])
        prob = Problem(Minimize(norm(self.x)), [self.x == 0])
        prob.unpack_results(s.ECOS, results_dict)
        self.assertItemsAlmostEqual(self.x.value, [0, 0])
        self.assertAlmostEqual(prob.value, 0)
        self.assertAlmostEqual(prob.status, s.OPTIMAL)

        if s.CVXOPT in installed_solvers():
            import cvxopt
            prob = Problem(Minimize(norm(self.x)), [self.x == 0])
            args = prob.get_problem_data(s.CVXOPT)
            results_dict = cvxopt.solvers.conelp(args["c"], args["G"],
                                                 args["h"], args["dims"],
                                                 args["A"], args["b"])
            prob = Problem(Minimize(norm(self.x)), [self.x == 0])
            prob.unpack_results(s.CVXOPT, results_dict)
            self.assertItemsAlmostEqual(self.x.value, [0, 0])
            self.assertAlmostEqual(prob.value, 0)
            self.assertAlmostEqual(prob.status, s.OPTIMAL)

    # Test silencing and enabling solver messages.
    def test_verbose(self):
        import sys
        # From http://stackoverflow.com/questions/5136611/capture-stdout-from-a-script-in-python
        # setup the environment
        outputs = {True: [], False: []}
        backup = sys.stdout
        # ####
        for verbose in [True, False]:
            for solver in installed_solvers():
                # Don't test GLPK because there's a race
                # condition in setting CVXOPT solver options.
                if solver in ["GLPK", "GLPK_MI", "MOSEK", "CBC", "LS"]:
                    continue
                if solver == "ELEMENTAL":
                    # ELEMENTAL's stdout is separate from python,
                    # so we have to do this.
                    # Note: This probably breaks (badly) on Windows.
                    import os
                    import tempfile

                    stdout_fd = 1
                    tmp_handle = tempfile.TemporaryFile(bufsize=0)
                    os.dup2(tmp_handle.fileno(), stdout_fd)
                else:
                    sys.stdout = StringIO()  # capture output

                p = Problem(Minimize(self.a + self.x[0]),
                            [self.a >= 2, self.x >= 2])

                if SOLVERS[solver].MIP_CAPABLE:
                    p.constraints.append(Bool() == 0)
                    p.solve(verbose=verbose, solver=solver)

                if SOLVERS[solver].EXP_CAPABLE:
                    p = Problem(Minimize(self.a), [log(self.a) >= 2])
                    p.solve(verbose=verbose, solver=solver)

                if SOLVERS[solver].SDP_CAPABLE:
                    p = Problem(Minimize(self.a), [lambda_min(self.a) >= 2])
                    p.solve(verbose=verbose, solver=solver)

                if solver == "ELEMENTAL":
                    # ELEMENTAL's stdout is separate from python,
                    # so we have to do this.
                    tmp_handle.seek(0)
                    out = tmp_handle.read()
                    tmp_handle.close()
                else:
                    out = sys.stdout.getvalue()  # release output

                outputs[verbose].append((out, solver))
        # ####
        sys.stdout.close()  # close the stream
        sys.stdout = backup  # restore original stdout
        for output, solver in outputs[True]:
            print(solver)
            assert len(output) > 0
        for output, solver in outputs[False]:
            print(solver)
            assert len(output) == 0

    # Test registering other solve methods.
    def test_register_solve(self):
        Problem.register_solve("test", lambda self: 1)
        p = Problem(Minimize(1))
        result = p.solve(method="test")
        self.assertEqual(result, 1)

        def test(self, a, b=2):
            return (a, b)
        Problem.register_solve("test", test)
        p = Problem(Minimize(0))
        result = p.solve(1, b=3, method="test")
        self.assertEqual(result, (1, 3))
        result = p.solve(1, method="test")
        self.assertEqual(result, (1, 2))
        result = p.solve(1, method="test", b=4)
        self.assertEqual(result, (1, 4))

    def test_consistency(self):
        """Test that variables and constraints keep a consistent order.
        """
        import itertools
        num_solves = 4
        vars_lists = []
        ineqs_lists = []
        var_ids_order_created = []
        for k in range(num_solves):
            sum = 0
            constraints = []
            var_ids = []
            for i in range(100):
                var = Variable(name=str(i))
                var_ids.append(var.id)
                sum += var
                constraints.append(var >= i)
            var_ids_order_created.append(var_ids)
            obj = Minimize(sum)
            p = Problem(obj, constraints)
            objective, constraints = p.canonicalize()
            sym_data = SymData(objective, constraints, SOLVERS[s.ECOS])
            # Sort by offset.
            vars_ = sorted(sym_data.var_offsets.items(),
                           key=lambda key_val: key_val[1])
            vars_ = [var_id for (var_id, offset) in vars_]
            vars_lists.append(vars_)
            ineqs_lists.append(sym_data.constr_map[s.LEQ])

        # Verify order of variables is consistent.
        for i in range(num_solves):
            self.assertEqual(var_ids_order_created[i],
                             vars_lists[i])
        for i in range(num_solves):
            for idx, constr in enumerate(ineqs_lists[i]):
                var_id, _ = lu.get_expr_vars(constr.expr)[0]
                self.assertEqual(var_ids_order_created[i][idx],
                                 var_id)

    # Test removing duplicate constraint objects.
    def test_duplicate_constraints(self):
        eq = (self.x == 2)
        le = (self.x <= 2)
        obj = 0

        def test(self):
            objective, constraints = self.canonicalize()
            sym_data = SymData(objective, constraints, SOLVERS[s.CVXOPT])
            return (len(sym_data.constr_map[s.EQ]),
                    len(sym_data.constr_map[s.LEQ]))
        Problem.register_solve("test", test)
        p = Problem(Minimize(obj), [eq, eq, le, le])
        result = p.solve(method="test")
        self.assertEqual(result, (1, 1))

        # Internal constraints.
        X = Semidef(2)
        obj = sum_entries(X + X)
        p = Problem(Minimize(obj))
        result = p.solve(method="test")
        self.assertEqual(result, (0, 1))

        # Duplicates from non-linear constraints.
        exp = norm(self.x, 2)
        prob = Problem(Minimize(0), [exp <= 1, exp <= 2])
        result = prob.solve(method="test")
        self.assertEqual(result, (0, 4))

    # Test the is_dcp method.
    def test_is_dcp(self):
        p = Problem(Minimize(normInf(self.a)))
        self.assertEqual(p.is_dcp(), True)

        p = Problem(Maximize(normInf(self.a)))
        self.assertEqual(p.is_dcp(), False)
        with self.assertRaises(Exception) as cm:
            p.solve()
        self.assertEqual(str(cm.exception), "Problem does not follow DCP rules.")
        p.solve(ignore_dcp=True)

    # Test the is_qp method.
    def test_is_qp(self):
        A = numpy.random.randn(4, 3)
        b = numpy.random.randn(4)
        Aeq = numpy.random.randn(2,3)
        beq = numpy.random.randn(2)
        F = numpy.random.randn(2,3)
        g = numpy.random.randn(2)
        obj = sum_squares(A*self.y - b)
        p = Problem(Minimize(obj),[])
        self.assertEqual(p.is_qp(), True)

        p = Problem(Minimize(obj),[Aeq * self.y == beq, F * self.y <= g])
        self.assertEqual(p.is_qp(), True)

        p = Problem(Minimize(obj),[max_elemwise(1, 3 * self.y) <= 200, abs(2 * self.y) <= 100, 
            norm(2 * self.y, 1) <= 1000, Aeq * self.y == beq])
        self.assertEqual(p.is_qp(), True)

        p = Problem(Minimize(obj),[max_elemwise(1, 3 * self.y ** 2) <= 200])
        self.assertEqual(p.is_qp(), False)

    # Test problems involving variables with the same name.
    def test_variable_name_conflict(self):
        var = Variable(name='a')
        p = Problem(Maximize(self.a + var), [var == 2 + self.a, var <= 3])
        result = p.solve()
        self.assertAlmostEqual(result, 4.0)
        self.assertAlmostEqual(self.a.value, 1)
        self.assertAlmostEqual(var.value, 3)

    # Test adding problems
    def test_add_problems(self):
        prob1 = Problem(Minimize(self.a), [self.a >= self.b])
        prob2 = Problem(Minimize(2*self.b), [self.a >= 1, self.b >= 2])
        prob_minimize = prob1 + prob2
        self.assertEqual(len(prob_minimize.constraints), 3)
        self.assertAlmostEqual(prob_minimize.solve(), 6)
        prob3 = Problem(Maximize(self.a), [self.b <= 1])
        prob4 = Problem(Maximize(2*self.b), [self.a <= 2])
        prob_maximize = prob3 + prob4
        self.assertEqual(len(prob_maximize.constraints), 2)
        self.assertAlmostEqual(prob_maximize.solve(), 4)

        # Test using sum function
        prob5 = Problem(Minimize(3*self.a))
        prob_sum = sum([prob1, prob2, prob5])
        self.assertEqual(len(prob_sum.constraints), 3)
        self.assertAlmostEqual(prob_sum.solve(), 12)
        prob_sum = sum([prob1])
        self.assertEqual(len(prob_sum.constraints), 1)

        # Test Minimize + Maximize
        with self.assertRaises(DCPError) as cm:
            prob_bad_sum = prob1 + prob3
        self.assertEqual(str(cm.exception), "Problem does not follow DCP rules.")

    # Test problem multiplication by scalar
    def test_mul_problems(self):
        prob1 = Problem(Minimize(pow(self.a, 2)), [self.a >= 2])
        answer = prob1.solve()
        factors = [0, 1, 2.3, -4.321]
        for f in factors:
            self.assertAlmostEqual((f * prob1).solve(), f * answer)
            self.assertAlmostEqual((prob1 * f).solve(), f * answer)

    # Test problem linear combinations
    def test_lin_combination_problems(self):
        prob1 = Problem(Minimize(self.a), [self.a >= self.b])
        prob2 = Problem(Minimize(2*self.b), [self.a >= 1, self.b >= 2])
        prob3 = Problem(Maximize(-pow(self.b + self.a, 2)), [self.b >= 3])

        # simple addition and multiplication
        combo1 = prob1 + 2 * prob2
        combo1_ref = Problem(Minimize(self.a + 4 * self.b), [self.a >= self.b, self.a >= 1, self.b >= 2])
        self.assertAlmostEqual(combo1.solve(), combo1_ref.solve())

        # division and subtraction
        combo2 = prob1 - prob3/2
        combo2_ref = Problem(Minimize(self.a + pow(self.b + self.a, 2)/2), [self.b >= 3, self.a >= self.b])
        self.assertAlmostEqual(combo2.solve(), combo2_ref.solve())

        # multiplication with 0 (prob2's constraints should still hold)
        combo3 = prob1 + 0 * prob2 - 3 * prob3
        combo3_ref = Problem(Minimize(self.a + 3 * pow(self.b + self.a, 2)), [self.a >= self.b, self.a >= 1, self.b >= 3])
        self.assertAlmostEqual(combo3.solve(), combo3_ref.solve())

    # Test solving problems in parallel.
    def test_solve_parallel(self):
        p = Parameter()
        problem = Problem(Minimize(square(self.a) + square(self.b) + p),
                          [self.b >= 2, self.a >= 1])
        p.value = 1
        # Ensure that parallel solver still works after repeated calls
        for _ in range(2):
            result = problem.solve(parallel=True)
            self.assertAlmostEqual(result, 6.0)
            self.assertEqual(problem.status, s.OPTIMAL)
            self.assertAlmostEqual(self.a.value, 1)
            self.assertAlmostEqual(self.b.value, 2)
            self.a.value = 0
            self.b.value = 0
        # The constant p should not be a separate problem, but rather added to
        # the first separable problem.
        self.assertTrue(len(problem._separable_problems) == 2)

        # Ensure that parallel solver works with options.
        result = problem.solve(parallel=True, verbose=True, warm_start=True)
        self.assertAlmostEqual(result, 6.0)
        self.assertEqual(problem.status, s.OPTIMAL)
        self.assertAlmostEqual(self.a.value, 1)
        self.assertAlmostEqual(self.b.value, 2)

        # Ensure that parallel solver works when problem changes.
        problem.objective = Minimize(square(self.a) + square(self.b))
        result = problem.solve(parallel=True)
        self.assertAlmostEqual(result, 5.0)
        self.assertEqual(problem.status, s.OPTIMAL)
        self.assertAlmostEqual(self.a.value, 1)
        self.assertAlmostEqual(self.b.value, 2)

    # Test scalar LP problems.
    def test_scalar_lp(self):
        p = Problem(Minimize(3*self.a), [self.a >= 2])
        result = p.solve()
        self.assertAlmostEqual(result, 6)
        self.assertAlmostEqual(self.a.value, 2)

        p = Problem(Maximize(3*self.a - self.b),
                    [self.a <= 2, self.b == self.a, self.b <= 5])
        result = p.solve()
        self.assertAlmostEqual(result, 4.0)
        self.assertAlmostEqual(self.a.value, 2)
        self.assertAlmostEqual(self.b.value, 2)

        # With a constant in the objective.
        p = Problem(Minimize(3*self.a - self.b + 100),
                    [self.a >= 2,
                     self.b + 5*self.c - 2 == self.a,
                     self.b <= 5 + self.c])
        result = p.solve()
        self.assertAlmostEqual(result, 101 + 1.0/6)
        self.assertAlmostEqual(self.a.value, 2)
        self.assertAlmostEqual(self.b.value, 5-1.0/6)
        self.assertAlmostEqual(self.c.value, -1.0/6)

        # Test status and value.
        exp = Maximize(self.a)
        p = Problem(exp, [self.a <= 2])
        result = p.solve(solver=s.ECOS)
        self.assertEqual(result, p.value)
        self.assertEqual(p.status, s.OPTIMAL)
        assert self.a.value is not None
        assert p.constraints[0].dual_value is not None

        # Unbounded problems.
        p = Problem(Maximize(self.a), [self.a >= 2])
        p.solve(solver=s.ECOS)
        self.assertEqual(p.status, s.UNBOUNDED)
        assert numpy.isinf(p.value)
        assert p.value > 0
        assert self.a.value is None
        assert p.constraints[0].dual_value is None

        if s.CVXOPT in installed_solvers():
            p = Problem(Minimize(-self.a), [self.a >= 2])
            result = p.solve(solver=s.CVXOPT)
            self.assertEqual(result, p.value)
            self.assertEqual(p.status, s.UNBOUNDED)
            assert numpy.isinf(p.value)
            assert p.value < 0

        # Infeasible problems.
        p = Problem(Maximize(self.a), [self.a >= 2, self.a <= 1])
        self.a.save_value(2)
        p.constraints[0].save_value(2)

        result = p.solve(solver=s.ECOS)
        self.assertEqual(result, p.value)
        self.assertEqual(p.status, s.INFEASIBLE)
        assert numpy.isinf(p.value)
        assert p.value < 0
        assert self.a.value is None
        assert p.constraints[0].dual_value is None

        p = Problem(Minimize(-self.a), [self.a >= 2, self.a <= 1])
        result = p.solve(solver=s.ECOS)
        self.assertEqual(result, p.value)
        self.assertEqual(p.status, s.INFEASIBLE)
        assert numpy.isinf(p.value)
        assert p.value > 0

    # Test vector LP problems.
    def test_vector_lp(self):
        c = Constant(numpy.matrix([1, 2]).T).value
        p = Problem(Minimize(c.T*self.x), [self.x >= c])
        result = p.solve()
        self.assertAlmostEqual(result, 5)
        self.assertItemsAlmostEqual(self.x.value, [1, 2])

        A = Constant(numpy.matrix([[3, 5], [1, 2]]).T).value
        I = Constant([[1, 0], [0, 1]])
        p = Problem(Minimize(c.T*self.x + self.a),
                    [A*self.x >= [-1, 1],
                     4*I*self.z == self.x,
                     self.z >= [2, 2],
                     self.a >= 2])
        result = p.solve()
        self.assertAlmostEqual(result, 26, places=3)
        obj = c.T*self.x.value + self.a.value
        self.assertAlmostEqual(obj[0, 0], result)
        self.assertItemsAlmostEqual(self.x.value, [8, 8], places=3)
        self.assertItemsAlmostEqual(self.z.value, [2, 2], places=3)

    def test_ecos_noineq(self):
        """Test ECOS with no inequality constraints.
        """
        T = Constant(numpy.ones((2, 2))).value
        p = Problem(Minimize(1), [self.A == T])
        result = p.solve(solver=s.ECOS)
        self.assertAlmostEqual(result, 1)
        self.assertItemsAlmostEqual(self.A.value, T)

    # Test matrix LP problems.
    def test_matrix_lp(self):
        T = Constant(numpy.ones((2, 2))).value
        p = Problem(Minimize(1), [self.A == T])
        result = p.solve()
        self.assertAlmostEqual(result, 1)
        self.assertItemsAlmostEqual(self.A.value, T)

        T = Constant(numpy.ones((2, 3))*2).value
        c = Constant(numpy.matrix([3, 4]).T).value
        p = Problem(Minimize(1), [self.A >= T*self.C,
                                  self.A == self.B, self.C == T.T])
        result = p.solve()
        self.assertAlmostEqual(result, 1)
        self.assertItemsAlmostEqual(self.A.value, self.B.value)
        self.assertItemsAlmostEqual(self.C.value, T)
        assert (self.A.value >= T*self.C.value).all()

        # Test variables are dense.
        self.assertEqual(type(self.A.value), intf.DEFAULT_INTF.TARGET_MATRIX)

    # Test variable promotion.
    def test_variable_promotion(self):
        p = Problem(Minimize(self.a), [self.x <= self.a, self.x == [1, 2]])
        result = p.solve()
        self.assertAlmostEqual(result, 2)
        self.assertAlmostEqual(self.a.value, 2)

        p = Problem(Minimize(self.a),
                    [self.A <= self.a,
                     self.A == [[1, 2], [3, 4]]
                     ])
        result = p.solve()
        self.assertAlmostEqual(result, 4)
        self.assertAlmostEqual(self.a.value, 4)

        # Promotion must happen before the multiplication.
        p = Problem(Minimize([[1], [1]]*(self.x + self.a + 1)),
                    [self.a + self.x >= [1, 2]])
        result = p.solve()
        self.assertAlmostEqual(result, 5)

    # Test parameter promotion.
    def test_parameter_promotion(self):
        a = Parameter()
        exp = [[1, 2], [3, 4]]*a
        a.value = 2
        assert not (exp.value - 2*numpy.array([[1, 2], [3, 4]]).T).any()

    def test_parameter_problems(self):
        """Test problems with parameters.
        """
        p1 = Parameter()
        p2 = Parameter(3, sign="negative")
        p3 = Parameter(4, 4, sign="positive")
        p = Problem(Maximize(p1*self.a), [self.a + p1 <= p2, self.b <= p3 + p3 + 2])
        p1.value = 2
        p2.value = -numpy.ones((3, 1))
        p3.value = numpy.ones((4, 4))
        result = p.solve()
        self.assertAlmostEqual(result, -6)

        p1.value = None
        with self.assertRaises(Exception) as cm:
            p.solve()
        self.assertEqual(str(cm.exception), "Problem has missing parameter value.")

    def test_constraint_error(self):
        """Tests non-standard input for constraints.
        """
        with self.assertRaises(Exception) as cm:
            Problem(Maximize(self.a), self.a)
        self.assertEqual(str(cm.exception), "Problem constraints must be a list.")

    # Test problems with normInf
    def test_normInf(self):
        # Constant argument.
        p = Problem(Minimize(normInf(-2)))
        result = p.solve()
        self.assertAlmostEqual(result, 2)

        # Scalar arguments.
        p = Problem(Minimize(normInf(self.a)), [self.a >= 2])
        result = p.solve()
        self.assertAlmostEqual(result, 2)
        self.assertAlmostEqual(self.a.value, 2)

        p = Problem(Minimize(3*normInf(self.a + 2*self.b) + self.c),
                    [self.a >= 2, self.b <= -1, self.c == 3])
        result = p.solve()
        self.assertAlmostEqual(result, 3)
        self.assertAlmostEqual(self.a.value + 2*self.b.value, 0)
        self.assertAlmostEqual(self.c.value, 3)

        # Maximize
        p = Problem(Maximize(-normInf(self.a)), [self.a <= -2])
        result = p.solve()
        self.assertAlmostEqual(result, -2)
        self.assertAlmostEqual(self.a.value, -2)

        # Vector arguments.
        p = Problem(Minimize(normInf(self.x - self.z) + 5),
                    [self.x >= [2, 3], self.z <= [-1, -4]])
        result = p.solve()
        self.assertAlmostEqual(float(result), 12)
        self.assertAlmostEqual(float(list(self.x.value)[1] - list(self.z.value)[1]), 7)

    # Test problems with norm1
    def test_norm1(self):
        # Constant argument.
        p = Problem(Minimize(norm1(-2)))
        result = p.solve()
        self.assertAlmostEqual(result, 2)

        # Scalar arguments.
        p = Problem(Minimize(norm1(self.a)), [self.a <= -2])
        result = p.solve()
        self.assertAlmostEqual(result, 2)
        self.assertAlmostEqual(self.a.value, -2)

        # Maximize
        p = Problem(Maximize(-norm1(self.a)), [self.a <= -2])
        result = p.solve()
        self.assertAlmostEqual(result, -2)
        self.assertAlmostEqual(self.a.value, -2)

        # Vector arguments.
        p = Problem(Minimize(norm1(self.x - self.z) + 5),
                    [self.x >= [2, 3], self.z <= [-1, -4]])
        result = p.solve()
        self.assertAlmostEqual(float(result), 15)
        self.assertAlmostEqual(float(list(self.x.value)[1] - list(self.z.value)[1]), 7)

    # Test problems with norm2
    def test_norm2(self):
        # Constant argument.
        p = Problem(Minimize(norm2(-2)))
        result = p.solve()
        self.assertAlmostEqual(result, 2)

        # Scalar arguments.
        p = Problem(Minimize(norm2(self.a)), [self.a <= -2])
        result = p.solve()
        self.assertAlmostEqual(result, 2)
        self.assertAlmostEqual(self.a.value, -2)

        # Maximize
        p = Problem(Maximize(-norm2(self.a)), [self.a <= -2])
        result = p.solve()
        self.assertAlmostEqual(result, -2)
        self.assertAlmostEqual(self.a.value, -2)

        # Vector arguments.
        p = Problem(Minimize(norm2(self.x - self.z) + 5),
                    [self.x >= [2, 3], self.z <= [-1, -4]])
        result = p.solve()
        self.assertAlmostEqual(result, 12.61577)
        self.assertItemsAlmostEqual(self.x.value, [2, 3])
        self.assertItemsAlmostEqual(self.z.value, [-1, -4])

        # Row  arguments.
        p = Problem(Minimize(norm2((self.x - self.z).T) + 5),
                    [self.x >= [2, 3], self.z <= [-1, -4]])
        result = p.solve()
        self.assertAlmostEqual(result, 12.61577)
        self.assertItemsAlmostEqual(self.x.value, [2, 3])
        self.assertItemsAlmostEqual(self.z.value, [-1, -4])

    # Test problems with abs
    def test_abs(self):
        p = Problem(Minimize(sum_entries(abs(self.A))), [-2 >= self.A])
        result = p.solve()
        self.assertAlmostEqual(result, 8)
        self.assertItemsAlmostEqual(self.A.value, [-2, -2, -2, -2])

    # Test problems with quad_form.
    def test_quad_form(self):
        with self.assertRaises(Exception) as cm:
            Problem(Minimize(quad_form(self.x, self.A))).solve()
        self.assertEqual(str(cm.exception), "At least one argument to quad_form must be constant.")

        with self.assertRaises(Exception) as cm:
            Problem(Minimize(quad_form(1, self.A))).solve()
        self.assertEqual(str(cm.exception), "Invalid dimensions for arguments.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises(Exception) as cm:
                Problem(Minimize(quad_form(self.x, [[-1, 0], [0, 9]]))).solve()
            self.assertEqual(str(cm.exception), "Problem does not follow DCP rules.")

        P = [[4, 0], [0, 9]]
        p = Problem(Minimize(quad_form(self.x, P)), [self.x >= 1])
        result = p.solve()
        self.assertAlmostEqual(result, 13, places=3)

        c = [1, 2]
        p = Problem(Minimize(quad_form(c, self.A)), [self.A >= 1])
        result = p.solve()
        self.assertAlmostEqual(result, 9)

        c = [1, 2]
        P = [[4, 0], [0, 9]]
        p = Problem(Minimize(quad_form(c, P)))
        result = p.solve()
        self.assertAlmostEqual(result, 40)

    # Test combining atoms
    def test_mixed_atoms(self):
        p = Problem(Minimize(norm2(5 + norm1(self.z)
                                   + norm1(self.x) +
                                   normInf(self.x - self.z))),
                    [self.x >= [2, 3], self.z <= [-1, -4], norm2(self.x + self.z) <= 2])
        result = p.solve()
        self.assertAlmostEqual(result, 22)
        self.assertItemsAlmostEqual(self.x.value, [2, 3])
        self.assertItemsAlmostEqual(self.z.value, [-1, -4])

    # Test multiplying by constant atoms.
    def test_mult_constant_atoms(self):
        p = Problem(Minimize(norm2([3, 4])*self.a), [self.a >= 2])
        result = p.solve()
        self.assertAlmostEqual(result, 10)
        self.assertAlmostEqual(self.a.value, 2)

    # Test recovery of dual variables.
    def test_dual_variables(self):
        for solver in [s.ECOS, s.SCS, s.CVXOPT]:
            if solver in installed_solvers():
                if solver == s.SCS:
                    acc = 1
                else:
                    acc = 5
                p = Problem(Minimize(norm1(self.x + self.z)),
                            [self.x >= [2, 3],
                            [[1, 2], [3, 4]]*self.z == [-1, -4],
                            norm2(self.x + self.z) <= 100])
                result = p.solve(solver=solver)
                self.assertAlmostEqual(result, 4, places=acc)
                self.assertItemsAlmostEqual(self.x.value, [4, 3], places=acc)
                self.assertItemsAlmostEqual(self.z.value, [-4, 1], places=acc)
                # Dual values
                self.assertItemsAlmostEqual(p.constraints[0].dual_value, [0, 1], places=acc)
                self.assertItemsAlmostEqual(p.constraints[1].dual_value, [-1, 0.5], places=acc)
                self.assertAlmostEqual(p.constraints[2].dual_value, 0, places=acc)

                T = numpy.ones((2, 3))*2
                c = numpy.matrix([3, 4]).T
                p = Problem(Minimize(1),
                            [self.A >= T*self.C,
                            self.A == self.B,
                            self.C == T.T])
                result = p.solve(solver=solver)
                # Dual values
                self.assertItemsAlmostEqual(p.constraints[0].dual_value, 4*[0], places=acc)
                self.assertItemsAlmostEqual(p.constraints[1].dual_value, 4*[0], places=acc)
                self.assertItemsAlmostEqual(p.constraints[2].dual_value, 6*[0], places=acc)

    # Test problems with indexing.
    def test_indexing(self):
        # Vector variables
        p = Problem(Maximize(self.x[0, 0]), [self.x[0, 0] <= 2, self.x[1, 0] == 3])
        result = p.solve()
        self.assertAlmostEqual(result, 2)
        self.assertItemsAlmostEqual(self.x.value, [2, 3])

        n = 10
        A = numpy.arange(n*n)
        A = numpy.reshape(A, (n, n))
        x = Variable(n, n)
        p = Problem(Minimize(sum_entries(x)), [x == A])
        result = p.solve()
        answer = n*n*(n*n+1)/2 - n*n
        self.assertAlmostEqual(result, answer)

        # Matrix variables
        p = Problem(Maximize(sum(self.A[i, i] + self.A[i, 1-i] for i in range(2))),
                    [self.A <= [[1, -2], [-3, 4]]])
        result = p.solve()
        self.assertAlmostEqual(result, 0)
        self.assertItemsAlmostEqual(self.A.value, [1, -2, -3, 4])

        # Indexing arithmetic expressions.
        exp = [[1, 2], [3, 4]]*self.z + self.x
        p = Problem(Minimize(exp[1, 0]), [self.x == self.z, self.z == [1, 2]])
        result = p.solve()
        self.assertAlmostEqual(result, 12)
        self.assertItemsAlmostEqual(self.x.value, self.z.value)

    def test_non_python_int_index(self):
        """Test problems that have special types as indices.
        """
        import sys
        if sys.version_info > (3,):
            my_long = int
        else:
            my_long = long
        # Test with long indices.
        cost = self.x[0:my_long(2)][0]
        p = Problem(Minimize(cost), [self.x == 1])
        result = p.solve()
        self.assertAlmostEqual(result, 1)
        self.assertItemsAlmostEqual(self.x.value, [1, 1])

        # Test with numpy64 indices.
        cost = self.x[0:numpy.int64(2)][0]
        p = Problem(Minimize(cost), [self.x == 1])
        result = p.solve()
        self.assertAlmostEqual(result, 1)
        self.assertItemsAlmostEqual(self.x.value, [1, 1])

        # Test with float.
        cost = self.x[numpy.float32(0)]
        p = Problem(Minimize(cost), [self.x == 1])
        result = p.solve()
        self.assertAlmostEqual(result, 1)
        self.assertItemsAlmostEqual(self.x.value, [1, 1])

    # Test problems with slicing.
    def test_slicing(self):
        p = Problem(Maximize(sum_entries(self.C)), [self.C[1:3, :] <= 2, self.C[0, :] == 1])
        result = p.solve()
        self.assertAlmostEqual(result, 10)
        self.assertItemsAlmostEqual(self.C.value, 2*[1, 2, 2])

        p = Problem(Maximize(sum_entries(self.C[0:3:2, 1])),
                    [self.C[1:3, :] <= 2, self.C[0, :] == 1])
        result = p.solve()
        self.assertAlmostEqual(result, 3)
        self.assertItemsAlmostEqual(self.C.value[0:3:2, 1], [1, 2])

        p = Problem(Maximize(sum_entries((self.C[0:2, :] + self.A)[:, 0:2])),
                    [self.C[1:3, :] <= 2, self.C[0, :] == 1,
                     (self.A + self.B)[:, 0] == 3, (self.A + self.B)[:, 1] == 2,
                     self.B == 1])
        result = p.solve()
        self.assertAlmostEqual(result, 12)
        self.assertItemsAlmostEqual(self.C.value[0:2, :], [1, 2, 1, 2])
        self.assertItemsAlmostEqual(self.A.value, [2, 2, 1, 1])

        p = Problem(Maximize([[3], [4]]*(self.C[0:2, :] + self.A)[:, 0]),
                    [self.C[1:3, :] <= 2, self.C[0, :] == 1,
                     [[1], [2]]*(self.A + self.B)[:, 0] == 3, (self.A + self.B)[:, 1] == 2,
                     self.B == 1, 3*self.A[:, 0] <= 3])
        result = p.solve()
        self.assertAlmostEqual(result, 12)
        self.assertItemsAlmostEqual(self.C.value[0:2, 0], [1, 2])
        self.assertItemsAlmostEqual(self.A.value, [1, -.5, 1, 1])

        p = Problem(Minimize(norm2((self.C[0:2, :] + self.A)[:, 0])),
                    [self.C[1:3, :] <= 2, self.C[0, :] == 1,
                     (self.A + self.B)[:, 0] == 3, (self.A + self.B)[:, 1] == 2,
                     self.B == 1])
        result = p.solve()
        self.assertAlmostEqual(result, 3)
        self.assertItemsAlmostEqual(self.C.value[0:2, 0], [1, -2])
        self.assertItemsAlmostEqual(self.A.value, [2, 2, 1, 1])

        # Transpose of slice.
        p = Problem(Maximize(sum_entries(self.C)), [self.C[1:3, :].T <= 2, self.C[0, :].T == 1])
        result = p.solve()
        self.assertAlmostEqual(result, 10)
        self.assertItemsAlmostEqual(self.C.value, 2*[1, 2, 2])

    # Test the vstack atom.
    def test_vstack(self):
        c = numpy.ones((1, 5))
        p = Problem(Minimize(c * vstack(self.x, self.y)),
                    [self.x == [1, 2],
                     self.y == [3, 4, 5]])
        result = p.solve()
        self.assertAlmostEqual(result, 15)

        c = numpy.ones((1, 4))
        p = Problem(Minimize(c * vstack(self.x, self.x)),
                    [self.x == [1, 2]])
        result = p.solve()
        self.assertAlmostEqual(result, 6)

        c = numpy.ones((2, 2))
        p = Problem(Minimize(sum_entries(vstack(self.A, self.C))),
                    [self.A >= 2*c,
                     self.C == -2])
        result = p.solve()
        self.assertAlmostEqual(result, -4)

        c = numpy.ones((1, 2))
        p = Problem(Minimize(sum_entries(vstack(c*self.A, c*self.B))),
                    [self.A >= 2,
                     self.B == -2])
        result = p.solve()
        self.assertAlmostEqual(result, 0)

        c = numpy.matrix([1, -1]).T
        p = Problem(Minimize(c.T * vstack(square(self.a), sqrt(self.b))),
                    [self.a == 2,
                     self.b == 16])
        with self.assertRaises(Exception) as cm:
            p.solve()
        self.assertEqual(str(cm.exception), "Problem does not follow DCP rules.")

    # Test the hstack atom.
    def test_hstack(self):
        c = numpy.ones((1, 5))
        p = Problem(Minimize(c * hstack(self.x.T, self.y.T).T),
                    [self.x == [1, 2],
                     self.y == [3, 4, 5]])
        result = p.solve()
        self.assertAlmostEqual(result, 15)

        c = numpy.ones((1, 4))
        p = Problem(Minimize(c * hstack(self.x.T, self.x.T).T),
                    [self.x == [1, 2]])
        result = p.solve()
        self.assertAlmostEqual(result, 6)

        c = numpy.ones((2, 2))
        p = Problem(Minimize(sum_entries(hstack(self.A.T, self.C.T))),
                    [self.A >= 2*c,
                     self.C == -2])
        result = p.solve()
        self.assertAlmostEqual(result, -4)

        D = Variable(3, 3)
        expr = hstack(self.C, D)
        p = Problem(Minimize(expr[0, 1] + sum_entries(hstack(expr, expr))),
                    [self.C >= 0,
                     D >= 0, D[0, 0] == 2, self.C[0, 1] == 3])
        result = p.solve()
        self.assertAlmostEqual(result, 13)

        c = numpy.matrix([1, -1]).T
        p = Problem(Minimize(c.T * hstack(square(self.a).T, sqrt(self.b).T).T),
                    [self.a == 2,
                     self.b == 16])
        with self.assertRaises(Exception) as cm:
            p.solve()
        self.assertEqual(str(cm.exception), "Problem does not follow DCP rules.")

    def test_bad_objective(self):
        """Test using a cvxpy expression as an objective.
        """
        with self.assertRaises(Exception) as cm:
            Problem(self.x+2)
        self.assertEqual(str(cm.exception), "Problem objective must be Minimize or Maximize.")

    # Test variable transpose.
    def test_transpose(self):
        p = Problem(Minimize(sum_entries(self.x)), [self.x.T >= numpy.matrix([1, 2])])
        result = p.solve()
        self.assertAlmostEqual(result, 3)
        self.assertItemsAlmostEqual(self.x.value, [1, 2])

        p = Problem(Minimize(sum_entries(self.C)),
                    [numpy.matrix([1, 1])*self.C.T >= numpy.matrix([0, 1, 2])])
        result = p.solve()
        value = self.C.value

        constraints = [1*self.C[i, 0] + 1*self.C[i, 1] >= i for i in range(3)]
        p = Problem(Minimize(sum_entries(self.C)), constraints)
        result2 = p.solve()
        self.assertAlmostEqual(result, result2)
        self.assertItemsAlmostEqual(self.C.value, value)

        p = Problem(Minimize(self.A[0, 1] - self.A.T[1, 0]),
                    [self.A == [[1, 2], [3, 4]]])
        result = p.solve()
        self.assertAlmostEqual(result, 0)

        exp = (-self.x).T
        p = Problem(Minimize(sum_entries(self.x)), [(-self.x).T <= 1])
        result = p.solve()
        self.assertAlmostEqual(result, -2)

        c = numpy.matrix([1, -1]).T
        p = Problem(Minimize(max_elemwise(c.T, 2, 2 + c.T)[1]))
        result = p.solve()
        self.assertAlmostEqual(result, 2)

        c = numpy.matrix([[1, -1, 2], [1, -1, 2]]).T
        p = Problem(Minimize(sum_entries(max_elemwise(c, 2, 2 + c).T[:, 0])))
        result = p.solve()
        self.assertAlmostEqual(result, 6)

        c = numpy.matrix([[1, -1, 2], [1, -1, 2]]).T
        p = Problem(Minimize(sum_entries(square(c.T).T[:, 0])))
        result = p.solve()
        self.assertAlmostEqual(result, 6)

        # Slice of transpose.
        p = Problem(Maximize(sum_entries(self.C)), [self.C.T[:, 1:3] <= 2, self.C.T[:, 0] == 1])
        result = p.solve()
        self.assertAlmostEqual(result, 10)
        self.assertItemsAlmostEqual(self.C.value, 2*[1, 2, 2])

    # Test multiplication on the left by a non-constant.
    def test_multiplication_on_left(self):
        c = numpy.matrix([1, 2]).T
        p = Problem(Minimize(c.T*self.A*c), [self.A >= 2])
        result = p.solve()
        self.assertAlmostEqual(result, 18)

        p = Problem(Minimize(self.a*2), [self.a >= 2])
        result = p.solve()
        self.assertAlmostEqual(result, 4)

        p = Problem(Minimize(self.x.T*c), [self.x >= 2])
        result = p.solve()
        self.assertAlmostEqual(result, 6)

        p = Problem(Minimize((self.x.T + self.z.T)*c),
                    [self.x >= 2, self.z >= 1])
        result = p.solve()
        self.assertAlmostEqual(result, 9)

    # Test redundant constraints in cvxopt.
    def test_redundant_constraints(self):
        obj = Minimize(sum_entries(self.x))
        constraints = [self.x == 2, self.x == 2, self.x.T == 2, self.x[0] == 2]
        p = Problem(obj, constraints)
        result = p.solve(solver=s.ECOS)
        self.assertAlmostEqual(result, 4)

        obj = Minimize(sum_entries(square(self.x)))
        constraints = [self.x == self.x]
        p = Problem(obj, constraints)
        result = p.solve(solver=s.ECOS)
        self.assertAlmostEqual(result, 0)

    # Test that symmetry is enforced.
    def test_sdp_symmetry(self):
        # TODO should these raise exceptions?
        # with self.assertRaises(Exception) as cm:
        #     lambda_max([[1,2],[3,4]])
        # self.assertEqual(str(cm.exception), "lambda_max called on non-symmetric matrix.")

        # with self.assertRaises(Exception) as cm:
        #     lambda_min([[1,2],[3,4]])
        # self.assertEqual(str(cm.exception), "lambda_min called on non-symmetric matrix.")

        p = Problem(Minimize(lambda_max(self.A)), [self.A >= 2])
        result = p.solve()
        self.assertItemsAlmostEqual(self.A.value, self.A.value.T, places=3)

        p = Problem(Minimize(lambda_max(self.A)), [self.A == [[1, 2], [3, 4]]])
        result = p.solve()
        self.assertEqual(p.status, s.INFEASIBLE)

    # Test SDP
    def test_sdp(self):
        # Ensure sdp constraints enforce transpose.
        obj = Maximize(self.A[1, 0] - self.A[0, 1])
        p = Problem(obj, [lambda_max(self.A) <= 100,
                          self.A[0, 0] == 2,
                          self.A[1, 1] == 2,
                          self.A[1, 0] == 2])
        result = p.solve()
        self.assertAlmostEqual(result, 0, places=3)

    # Test getting values for expressions.
    def test_expression_values(self):
        diff_exp = self.x - self.z
        inf_exp = normInf(diff_exp)
        sum_entries_exp = 5 + norm1(self.z) + norm1(self.x) + inf_exp
        constr_exp = norm2(self.x + self.z)
        obj = norm2(sum_entries_exp)
        p = Problem(Minimize(obj),
                    [self.x >= [2, 3], self.z <= [-1, -4], constr_exp <= 2])
        result = p.solve()
        self.assertAlmostEqual(result, 22)
        self.assertItemsAlmostEqual(self.x.value, [2, 3])
        self.assertItemsAlmostEqual(self.z.value, [-1, -4])
        # Expression values.
        self.assertItemsAlmostEqual(diff_exp.value, self.x.value - self.z.value)
        self.assertAlmostEqual(inf_exp.value,
                               LA.norm(self.x.value - self.z.value, numpy.inf))
        self.assertAlmostEqual(sum_entries_exp.value,
                               5 + LA.norm(self.z.value, 1) + LA.norm(self.x.value, 1) +
                               LA.norm(self.x.value - self.z.value, numpy.inf))
        self.assertAlmostEqual(constr_exp.value,
                               LA.norm(self.x.value + self.z.value, 2))
        self.assertAlmostEqual(obj.value, result)

    def test_mult_by_zero(self):
        """Test multiplication by zero.
        """
        exp = 0*self.a
        self.assertEqual(exp.value, 0)
        obj = Minimize(exp)
        p = Problem(obj)
        result = p.solve()
        self.assertAlmostEqual(result, 0)
        assert self.a.value is not None

    def test_div(self):
        """Tests a problem with division.
        """
        obj = Minimize(normInf(self.A/5))
        p = Problem(obj, [self.A >= 5])
        result = p.solve()
        self.assertAlmostEqual(result, 1)

    def test_mul_elemwise(self):
        """Tests problems with mul_elemwise.
        """
        c = [[1, -1], [2, -2]]
        expr = mul_elemwise(c, self.A)
        obj = Minimize(normInf(expr))
        p = Problem(obj, [self.A == 5])
        result = p.solve()
        self.assertAlmostEqual(result, 10)
        self.assertItemsAlmostEqual(expr.value, [5, -5] + [10, -10])

        # Test with a sparse matrix.
        import cvxopt
        interface = intf.get_matrix_interface(cvxopt.spmatrix)
        c = interface.const_to_matrix([1, 2])
        expr = mul_elemwise(c, self.x)
        obj = Minimize(normInf(expr))
        p = Problem(obj, [self.x == 5])
        result = p.solve()
        self.assertAlmostEqual(result, 10)
        self.assertItemsAlmostEqual(expr.value, [5, 10])

        # Test promotion.
        c = [[1, -1], [2, -2]]
        expr = mul_elemwise(c, self.a)
        obj = Minimize(normInf(expr))
        p = Problem(obj, [self.a == 5])
        result = p.solve()
        self.assertAlmostEqual(result, 10)
        self.assertItemsAlmostEqual(expr.value, [5, -5] + [10, -10])

    def test_invalid_solvers(self):
        """Tests that errors occur when you use an invalid solver.
        """
        with self.assertRaises(Exception) as cm:
            Problem(Minimize(Bool())).solve(solver=s.ECOS)

        expected = (
            "The solver ECOS cannot solve the problem "
            "because it cannot solve mixed-integer problems."
        )
        self.assertEqual(str(cm.exception), expected)

        with self.assertRaises(Exception) as cm:
            Problem(Minimize(lambda_max(self.a))).solve(solver=s.ECOS)

        expected = (
            "The solver ECOS cannot solve the problem "
            "because it cannot solve semidefinite problems."
        )
        self.assertEqual(str(cm.exception), expected)

        with self.assertRaises(Exception) as cm:
            Problem(Minimize(self.a)).solve(solver=s.SCS)

        expected = (
            "The solver SCS cannot solve the problem "
            "because it cannot solve unconstrained problems."
        )
        self.assertEqual(str(cm.exception), expected)

    def test_reshape(self):
        """Tests problems with reshape.
        """
        # Test on scalars.
        self.assertEqual(reshape(1, 1, 1).value, 1)

        # Test vector to matrix.
        x = Variable(4)
        mat = numpy.matrix([[1, -1], [2, -2]]).T
        vec = numpy.matrix([1, 2, 3, 4]).T
        vec_mat = numpy.matrix([[1, 2], [3, 4]]).T
        expr = reshape(x, 2, 2)
        obj = Minimize(sum_entries(mat*expr))
        prob = Problem(obj, [x == vec])
        result = prob.solve()
        self.assertAlmostEqual(result, numpy.sum(mat*vec_mat))

        # Test on matrix to vector.
        c = [1, 2, 3, 4]
        expr = reshape(self.A, 4, 1)
        obj = Minimize(expr.T*c)
        constraints = [self.A == [[-1, -2], [3, 4]]]
        prob = Problem(obj, constraints)
        result = prob.solve()
        self.assertAlmostEqual(result, 20)
        self.assertItemsAlmostEqual(expr.value, [-1, -2, 3, 4])
        self.assertItemsAlmostEqual(reshape(expr, 2, 2).value, [-1, -2, 3, 4])

        # Test matrix to matrix.
        expr = reshape(self.C, 2, 3)
        mat = numpy.matrix([[1, -1], [2, -2]])
        C_mat = numpy.matrix([[1, 4], [2, 5], [3, 6]])
        obj = Minimize(sum_entries(mat*expr))
        prob = Problem(obj, [self.C == C_mat])
        result = prob.solve()
        reshaped = numpy.reshape(C_mat, (2, 3), 'F')
        self.assertAlmostEqual(result, (mat.dot(reshaped)).sum())
        self.assertItemsAlmostEqual(expr.value, C_mat)

        # Test promoted expressions.
        c = numpy.matrix([[1, -1], [2, -2]]).T
        expr = reshape(c*self.a, 1, 4)
        obj = Minimize(expr*[1, 2, 3, 4])
        prob = Problem(obj, [self.a == 2])
        result = prob.solve()
        self.assertAlmostEqual(result, -6)
        self.assertItemsAlmostEqual(expr.value, 2*c)

        expr = reshape(c*self.a, 4, 1)
        obj = Minimize(expr.T*[1, 2, 3, 4])
        prob = Problem(obj, [self.a == 2])
        result = prob.solve()
        self.assertAlmostEqual(result, -6)
        self.assertItemsAlmostEqual(expr.value, 2*c)

    def test_vec(self):
        """Tests problems with vec.
        """
        c = [1, 2, 3, 4]
        expr = vec(self.A)
        obj = Minimize(expr.T*c)
        constraints = [self.A == [[-1, -2], [3, 4]]]
        prob = Problem(obj, constraints)
        result = prob.solve()
        self.assertAlmostEqual(result, 20)
        self.assertItemsAlmostEqual(expr.value, [-1, -2, 3, 4])

    def test_diag_prob(self):
        """Test a problem with diag.
        """
        C = Variable(3, 3)
        obj = Maximize(C[0, 2])
        constraints = [diag(C) == 1,
                       C[0, 1] == 0.6,
                       C[1, 2] == -0.3,
                       C == Semidef(3)]
        prob = Problem(obj, constraints)
        result = prob.solve()
        self.assertAlmostEqual(result, 0.583151, places=2)

    def test_presolve_constant_constraints(self):
        """Test that the presolver removes constraints with no variables.
        """
        x = Variable()
        obj = Maximize(sqrt(x))
        prob = Problem(obj, [Constant(2) <= 2])
        data = prob.get_problem_data(s.ECOS)
        A = data["A"]
        G = data["G"]
        for row in range(A.shape[0]):
            assert A[row, :].nnz > 0
        for row in range(G.shape[0]):
            assert G[row, :].nnz > 0

    def test_presolve_parameters(self):
        """Test presolve with parameters.
        """
        # Test with parameters.
        gamma = Parameter(sign="positive")
        x = Variable()
        obj = Minimize(x)
        prob = Problem(obj, [gamma == 1, x >= 0])
        gamma.value = 0
        prob.solve(solver=s.SCS)
        self.assertEqual(prob.status, s.INFEASIBLE)

        gamma.value = 1
        prob.solve(solver=s.SCS)
        self.assertEqual(prob.status, s.OPTIMAL)

    def test_parameter_expressions(self):
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
        obj = abs(x - 1)
        prob = Problem(Minimize(obj), [g == 0])
        prob.solve()
        x0.value = 1
        prob.solve()
        self.assertAlmostEqual(g.value, 0)

        # Test multiplication.
        prob = Problem(Minimize(x0*x), [x == 1])
        x0.value = 2
        prob.solve()
        x0.value = 1
        prob.solve()
        self.assertAlmostEqual(prob.value, 1)

    def test_change_constraints(self):
        """Test interaction of caching with changing constraints.
        """
        prob = Problem(Minimize(self.a), [self.a == 2, self.a >= 1])
        prob.solve()
        self.assertAlmostEqual(prob.value, 2)

        prob.constraints[0] = (self.a == 1)
        prob.solve()
        self.assertAlmostEqual(prob.value, 1)

    def test_psd_constraints(self):
        """Test positive definite constraints.
        """
        C = Variable(3, 3)
        obj = Maximize(C[0, 2])
        constraints = [diag(C) == 1,
                       C[0, 1] == 0.6,
                       C[1, 2] == -0.3,
                       C == C.T,
                       C >> 0]
        prob = Problem(obj, constraints)
        result = prob.solve()
        self.assertAlmostEqual(result, 0.583151, places=2)

        C = Variable(2, 2)
        obj = Maximize(C[0, 1])
        constraints = [C == 1, C >> [[2, 0], [0, 2]]]
        prob = Problem(obj, constraints)
        result = prob.solve()
        self.assertEqual(prob.status, s.INFEASIBLE)

        C = Symmetric(2, 2)
        obj = Minimize(C[0, 0])
        constraints = [C << [[2, 0], [0, 2]]]
        prob = Problem(obj, constraints)
        result = prob.solve()
        self.assertEqual(prob.status, s.UNBOUNDED)

    def test_psd_duals(self):
        """Test the duals of PSD constraints.
        """
        if s.CVXOPT in installed_solvers():
            # Test the dual values with cvxopt.
            C = Symmetric(2, 2)
            obj = Maximize(C[0, 0])
            constraints = [C << [[2, 0], [0, 2]]]
            prob = Problem(obj, constraints)
            result = prob.solve(solver=s.CVXOPT)
            self.assertAlmostEqual(result, 2)

            psd_constr_dual = constraints[0].dual_value
            C = Symmetric(2, 2)
            X = Semidef(2)
            obj = Maximize(C[0, 0])
            constraints = [X == [[2, 0], [0, 2]] - C]
            prob = Problem(obj, constraints)
            result = prob.solve(solver=s.CVXOPT)
            self.assertItemsAlmostEqual(constraints[0].dual_value, psd_constr_dual)

        # Test the dual values with SCS.
        C = Symmetric(2, 2)
        obj = Maximize(C[0, 0])
        constraints = [C << [[2, 0], [0, 2]]]
        prob = Problem(obj, constraints)
        result = prob.solve(solver=s.SCS)
        self.assertAlmostEqual(result, 2, places=4)

        psd_constr_dual = constraints[0].dual_value
        C = Symmetric(2, 2)
        X = Semidef(2)
        obj = Maximize(C[0, 0])
        constraints = [X == [[2, 0], [0, 2]] - C]
        prob = Problem(obj, constraints)
        result = prob.solve(solver=s.SCS)
        self.assertItemsAlmostEqual(constraints[0].dual_value, psd_constr_dual)

        # Test dual values with SCS that have off-diagonal entries.
        C = Symmetric(2, 2)
        obj = Maximize(C[0, 1] + C[1, 0])
        constraints = [C << [[2, 0], [0, 2]], C >= 0]
        prob = Problem(obj, constraints)
        result = prob.solve(solver=s.SCS)
        self.assertAlmostEqual(result, 4, places=3)

        psd_constr_dual = constraints[0].dual_value
        C = Symmetric(2, 2)
        X = Semidef(2)
        obj = Maximize(C[0, 1] + C[1, 0])
        constraints = [X == [[2, 0], [0, 2]] - C, C >= 0]
        prob = Problem(obj, constraints)
        result = prob.solve(solver=s.SCS)
        self.assertItemsAlmostEqual(constraints[0].dual_value, psd_constr_dual,
                                    places=3)

    def test_geo_mean(self):
        import numpy as np

        x = Variable(2)
        cost = geo_mean(x)
        prob = Problem(Maximize(cost), [x <= 1])
        prob.solve()
        self.assertAlmostEqual(prob.value, 1)

        prob = Problem(Maximize(cost), [sum(x) <= 1])
        prob.solve()
        self.assertItemsAlmostEqual(x.value, [.5, .5])

        x = Variable(3, 3)
        self.assertRaises(ValueError, geo_mean, x)

        x = Variable(3, 1)
        g = geo_mean(x)
        self.assertSequenceEqual(g.w, [Fraction(1, 3)]*3)

        x = Variable(1, 5)
        g = geo_mean(x)
        self.assertSequenceEqual(g.w, [Fraction(1, 5)]*5)

        # check that we get the right answer for
        # max geo_mean(x) s.t. sum(x) <= 1
        p = np.array([.07, .12, .23, .19, .39])

        def short_geo_mean(x, p):
            p = np.array(p)/sum(p)
            x = np.array(x)
            return np.prod(x**p)

        x = Variable(5)
        prob = Problem(Maximize(geo_mean(x, p)), [sum(x) <= 1])
        prob.solve()
        x = np.array(x.value).flatten()
        x_true = p/sum(p)

        self.assertTrue(np.allclose(prob.value, geo_mean(list(x), p).value))
        self.assertTrue(np.allclose(prob.value, short_geo_mean(x, p)))
        self.assertTrue(np.allclose(x, x_true, 1e-3))

        # check that we get the right answer for
        # max geo_mean(x) s.t. norm(x) <= 1
        x = Variable(5)
        prob = Problem(Maximize(geo_mean(x, p)), [norm(x) <= 1])
        prob.solve()
        x = np.array(x.value).flatten()
        x_true = np.sqrt(p/sum(p))

        self.assertTrue(np.allclose(prob.value, geo_mean(list(x), p).value))
        self.assertTrue(np.allclose(prob.value, short_geo_mean(x, p)))
        self.assertTrue(np.allclose(x, x_true, 1e-3))

        # the following 3 tests check vstack and hstack input to geo_mean
        # the following 3 formulations should be equivalent
        n = 5
        x_true = np.ones(n)
        x = Variable(n)

        Problem(Maximize(geo_mean(x)), [x <= 1]).solve()
        xval = np.array(x.value).flatten()
        self.assertTrue(np.allclose(xval, x_true, 1e-3))

        y = vstack(*[x[i] for i in range(n)])
        Problem(Maximize(geo_mean(y)), [x <= 1]).solve()
        xval = np.array(x.value).flatten()
        self.assertTrue(np.allclose(xval, x_true, 1e-3))

        y = hstack(*[x[i] for i in range(n)])
        Problem(Maximize(geo_mean(y)), [x <= 1]).solve()
        xval = np.array(x.value).flatten()
        self.assertTrue(np.allclose(xval, x_true, 1e-3))

    def test_pnorm(self):
        import numpy as np

        x = Variable(3, name='x')

        a = np.array([1.0, 2, 3])

        # todo: add -1, .5, .3, -2.3 and testing positivity constraints

        for p in (1, 1.6, 1.3, 2, 1.99, 3, 3.7, np.inf):
            prob = Problem(Minimize(pnorm(x, p=p)), [x.T*a >= 1])
            prob.solve()

            # formula is true for any a >= 0 with p > 1
            if p == np.inf:
                x_true = np.ones_like(a)/sum(a)
            elif p == 1:
                # only works for the particular a = [1,2,3]
                x_true = np.array([0, 0, 1.0/3])
            else:
                x_true = a**(1.0/(p-1))/a.dot(a**(1.0/(p-1)))

            x_alg = np.array(x.value).flatten()
            self.assertTrue(np.allclose(x_alg, x_true, 1e-3), 'p = {}'.format(p))
            self.assertTrue(np.allclose(prob.value, np.linalg.norm(x_alg, p)))
            self.assertTrue(np.allclose(np.linalg.norm(x_alg, p), pnorm(x_alg, p).value))

    def test_pnorm_concave(self):
        import numpy as np

        x = Variable(3, name='x')

        # test positivity constraints
        a = np.array([-1.0, 2, 3])
        for p in (-1, .5, .3, -2.3):
            prob = Problem(Minimize(sum_entries(abs(x-a))), [pnorm(x, p) >= 0])
            prob.solve()

            self.assertTrue(np.allclose(prob.value, 1))

        a = np.array([1.0, 2, 3])
        for p in (-1, .5, .3, -2.3):
            prob = Problem(Minimize(sum_entries(abs(x-a))), [pnorm(x, p) >= 0])
            prob.solve()

            self.assertTrue(np.allclose(prob.value, 0))

    def test_power(self):
        x = Variable()
        prob = Problem(Minimize(power(x, 1.7) + power(x, -2.3) - power(x, .45)))
        prob.solve()
        x = x.value
        self.assertTrue(__builtins__['abs'](1.7*x**.7 - 2.3*x**-3.3 - .45*x**-.55) <= 1e-3)

    def test_mul_elemwise(self):
        """Test a problem with mul_elemwise by a scalar.
        """
        import numpy as np
        T = 10
        J = 20
        rvec = np.random.randn(T, J)
        dy = np.random.randn(2*T, 1)
        theta = Variable(J)

        delta = 1e-3
        loglambda = rvec*theta  # rvec: TxJ regressor matrix, theta: (Jx1) cvx variable
        a = mul_elemwise(dy[0:T], loglambda)  # size(Tx1)
        b1 = exp(loglambda)
        b2 = mul_elemwise(delta, b1)
        cost = -a + b1

        cost = -a + b2  # size (Tx1)
        prob = Problem(Minimize(sum_entries(cost)))
        prob.solve(solver=s.SCS)

        obj = Minimize(sum_entries(mul_elemwise(2, self.x)))
        prob = Problem(obj, [self.x == 2])
        result = prob.solve()
        self.assertAlmostEqual(result, 8)
