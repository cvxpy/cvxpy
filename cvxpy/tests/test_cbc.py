from cvxpy import *
from cvxpy.tests.base_test import BaseTest


class TestSolvers(BaseTest):
    """ Unit tests for solver specific behavior. """

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

    def test_lp(self):
        """Tests basic LPs. (from test_elemental.py)
        """
        if CBC in installed_solvers():
            prob = Problem(Minimize(0), [self.x == 2])
            prob.solve(verbose=False, solver=CBC)
            self.assertAlmostEqual(prob.value, 0)
            self.assertItemsAlmostEqual(self.x.value, [2, 2])

            prob = Problem(Minimize(-self.a), [self.a <= 1])
            prob.solve(verbose=False, solver=CBC)
            self.assertAlmostEqual(prob.value, -1)
            self.assertAlmostEqual(self.a.value, 1)

    def test_lp_2(self):
        """Test a basic LP. (from test_solver.py::test_cvxopt_glpk)
        """
        # Either the problem is solved or CBC is not installed.
        if CBC in installed_solvers():
            prob = Problem(Minimize(norm(self.x, 1)), [self.x == 0])
            prob.solve(verbose=False, solver=CBC)
            self.assertAlmostEqual(prob.value, 0)
            self.assertItemsAlmostEqual(self.x.value, [0, 0])

            # Example from http://cvxopt.org/userguide/coneprog.html?highlight=solvers.lp#cvxopt.solvers.lp
            objective = Minimize(-4 * self.x[0] - 5 * self.x[1])
            constraints = [2 * self.x[0] + self.x[1] <= 3,
                           self.x[0] + 2 * self.x[1] <= 3,
                           self.x[0] >= 0,
                           self.x[1] >= 0]
            prob = Problem(objective, constraints)
            prob.solve(verbose=False, solver=CBC)
            self.assertAlmostEqual(prob.value, -9)
            self.assertItemsAlmostEqual(self.x.value, [1, 1])
        else:
            with self.assertRaises(Exception) as cm:
                prob = Problem(Minimize(norm(self.x, 1)), [self.x == 0])
                prob.solve(verbose=False, solver=CBC)
            self.assertEqual(str(cm.exception), "The solver %s is not installed." % CBC)

    def test_mip(self):
        """Test a basic MILP with CBC. (from test_solver.py::test_cvxopt_glpk_mi)
        """
        # Either the problem is solved or CBC is not installed.
        if CBC in installed_solvers():
            bool_var = Bool()
            int_var = Int()
            prob = Problem(Minimize(norm(self.x, 1)),
                           [self.x == bool_var, bool_var == 0])
            prob.solve(solver=CBC, verbose=False)
            self.assertAlmostEqual(prob.value, 0)
            self.assertAlmostEqual(bool_var.value, 0)
            self.assertItemsAlmostEqual(self.x.value, [0, 0])

            # Example from http://cvxopt.org/userguide/coneprog.html?highlight=solvers.lp#cvxopt.solvers.lp
            objective = Minimize(-4 * self.x[0] - 5 * self.x[1])
            constraints = [2 * self.x[0] + self.x[1] <= int_var,
                           self.x[0] + 2 * self.x[1] <= 3*bool_var,
                           self.x[0] >= 0,
                           self.x[1] >= 0,
                           int_var == 3*bool_var,
                           int_var == 3]
            prob = Problem(objective, constraints)
            prob.solve(solver=CBC, verbose=False)
            self.assertAlmostEqual(prob.value, -9)
            self.assertAlmostEqual(int_var.value, 3)
            self.assertAlmostEqual(bool_var.value, 1)
            self.assertItemsAlmostEqual(self.x.value, [1, 1])
        else:
            with self.assertRaises(Exception) as cm:
                prob = Problem(Minimize(norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=CBC, verbose=False)
            self.assertEqual(str(cm.exception), "The solver %s is not installed." % CBC)

    def test_hard_mip(self):
        """Test a hard knapsack problem with CBC.
        """
        # Either the problem is solved or CBC is not installed.
        if CBC in installed_solvers():
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

            X = Bool(n)
            prob = Problem(Maximize(sum_entries(mul_elemwise([i[1] for i in coeffs], X))),
                           [sum_entries(mul_elemwise([i[2] for i in coeffs], X)) <= c])
            prob.solve(verbose=False, solver=CBC)
            self.assertAlmostEqual(prob.value, z)  # objective
        else:
            with self.assertRaises(Exception) as cm:
                prob = Problem(Minimize(norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=CBC, verbose=False)
            self.assertEqual(str(cm.exception), "The solver %s is not installed." % CBC)

    def test_options(self):
        """Test that all the CBC solver options work.
        """
        if CBC in installed_solvers():
            prob = Problem(Minimize(norm(self.x, 1)), [self.x == Bool(2)])
            for i in range(2):
                # Some cut-generators seem to be buggy for now -> set to false
                prob.solve(solver=CBC, verbose=True, GomoryCuts=True, MIRCuts=True,
                           MIRCuts2=True, TwoMIRCuts=True, ResidualCapacityCuts=True,
                           KnapsackCuts=True, FlowCoverCuts=True, CliqueCuts=True,
                           LiftProjectCuts=True, AllDifferentCuts=False, OddHoleCuts=True,
                           RedSplitCuts=False, LandPCuts=False, PreProcessCuts=False,
                           ProbingCuts=True, SimpleRoundingCuts=True)
            self.assertItemsAlmostEqual(self.x.value, [0, 0])
        else:
            with self.assertRaises(Exception) as cm:
                prob.solve(solver=CBC)
                self.assertEqual(str(cm.exception), "The solver %s is not installed." % CBC)
