import unittest

import cvxpy as cp
from cvxpy.reductions.solvers.qp_solvers.osqp_qpif import OSQP
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS


class CustomQPSolverCalled(Exception):
    pass


class CustomConicSolverCalled(Exception):
    pass


class CustomQPSolver(OSQP):
    def name(self):
        return "CUSTOM_QP_SOLVER"

    def solve_via_data(self, *args, **kwargs):
        raise(CustomQPSolverCalled())


class CustomConicSolver(SCS):
    def name(self):
        return "CUSTOM_CONIC_SOLVER"

    def solve_via_data(self, *args, **kwargs):
        raise(CustomConicSolverCalled())


class ConflictingCustomSolver(OSQP):
    def name(self):
        return "OSQP"


class TestCustomSolvers(unittest.TestCase):
    def setUp(self):
        self.custom_qp_solver = CustomQPSolver()
        self.custom_conic_solver = CustomConicSolver()

    def test_custom_continuous_qp_solver_can_solve_continuous_qp(self):
        with self.assertRaises(CustomQPSolverCalled):
            self.solve_example_qp(solver=self.custom_qp_solver)

    def test_custom_mip_qp_solver_can_solve_mip_qp(self):
        self.custom_qp_solver.MIP_CAPABLE = True
        with self.assertRaises(CustomQPSolverCalled):
            self.solve_example_mixed_integer_qp(solver=self.custom_qp_solver)

    def test_custom_continuous_qp_solver_cannot_solve_mip_qp(self):
        self.custom_conic_solver.MIP_CAPABLE = False
        with self.assertRaises(cp.error.SolverError):
            self.solve_example_mixed_integer_qp(solver=self.custom_qp_solver)

    def test_custom_qp_solver_cannot_solve_socp(self):
        with self.assertRaises(cp.error.SolverError):
            self.solve_example_socp(solver=self.custom_qp_solver)

    def test_custom_continuous_conic_solver_can_solve_continuous_socp(self):
        with self.assertRaises(CustomConicSolverCalled):
            self.solve_example_socp(solver=self.custom_conic_solver)

    def test_custom_mip_conic_solver_can_solve_mip_socp(self):
        self.custom_conic_solver.MIP_CAPABLE = True
        with self.assertRaises(CustomConicSolverCalled):
            self.solve_example_mixed_integer_socp(solver=self.custom_conic_solver)

    def test_custom_continuous_conic_solver_cannot_solve_mip_socp(self):
        self.custom_conic_solver.MIP_CAPABLE = False
        with self.assertRaises(cp.error.SolverError):
            self.solve_example_mixed_integer_qp(solver=self.custom_conic_solver)

    def test_custom_conflicting_solver_fails(self):
        with self.assertRaises(cp.error.SolverError):
            self.solve_example_qp(solver=ConflictingCustomSolver())

    @staticmethod
    def solve_example_qp(solver):
        x = cp.Variable()
        quadratic = cp.sum_squares(x)
        problem = cp.Problem(cp.Minimize(quadratic))
        problem.solve(solver=solver)

    @staticmethod
    def solve_example_mixed_integer_qp(solver):
        x = cp.Variable()
        z = cp.Variable(integer=True)
        quadratic = cp.sum_squares(x + z)
        problem = cp.Problem(cp.Minimize(quadratic))
        problem.solve(solver=solver)

    @staticmethod
    def solve_example_socp(solver):
        x = cp.Variable(2)
        y = cp.Variable()
        quadratic = cp.sum_squares(x)
        problem = cp.Problem(cp.Minimize(quadratic), [cp.SOC(y, x)])
        problem.solve(solver=solver)

    @staticmethod
    def solve_example_mixed_integer_socp(solver):
        x = cp.Variable(2)
        y = cp.Variable()
        z = cp.Variable(integer=True)
        quadratic = cp.sum_squares(x + z)
        problem = cp.Problem(cp.Minimize(quadratic), [cp.SOC(y, x)])
        problem.solve(solver=solver)
