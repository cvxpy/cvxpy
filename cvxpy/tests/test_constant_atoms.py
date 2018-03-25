"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

# Tests atoms by calling them with a constant value.
from cvxpy.settings import (SCS, OSQP, ECOS, CVXOPT, GLPK, ELEMENTAL,
                            OPTIMAL, OPTIMAL_INACCURATE, ROBUST_KKTSOLVER, MOSEK)
from cvxpy.atoms import *
from cvxpy.problems.objective import *
from cvxpy.problems.problem import Problem
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.error import SolverError
import cvxpy.interface as intf
from cvxpy.reductions.solvers.solving_chain import construct_solving_chain
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
import numpy as np
import numpy.linalg as LA
import math
import itertools

ROBUST_CVXOPT = "robust_cvxopt"
SOLVER_TO_TOL = {SCS: 1e-2,
                 ECOS: 1e-7,
                 OSQP: 1e-1
}
SOLVERS_TO_TRY = [ECOS, SCS, OSQP]
# Test CVXOPT if installed.
if CVXOPT in INSTALLED_SOLVERS:
    SOLVERS_TO_TRY += [CVXOPT, ROBUST_CVXOPT]
    SOLVER_TO_TOL[CVXOPT] = 1e-7
    SOLVER_TO_TOL[ROBUST_CVXOPT] = 1e-7

# Test elemental if installed.
if ELEMENTAL in INSTALLED_SOLVERS:
    SOLVERS_TO_TRY.append(ELEMENTAL)
    SOLVER_TO_TOL[ELEMENTAL] = 1e-7

# Test MOSEK if installed.
if MOSEK in INSTALLED_SOLVERS:
    SOLVERS_TO_TRY.append(MOSEK)
    SOLVER_TO_TOL[MOSEK] = 1e-6

v_np = np.matrix([-1., 2, -2]).T

# Defined here to be used in KNOWN_SOLVER_ERRORS
log_sum_exp_axis_0 = lambda x: log_sum_exp(x, axis=0, keepdims=True)
log_sum_exp_axis_1 = lambda x: log_sum_exp(x, axis=1)

# Atom, solver pairs known to fail.
KNOWN_SOLVER_ERRORS = [
    # See https://github.com/cvxgrp/cvxpy/issues/249
    (log_sum_exp_axis_0, CVXOPT),
    (log_sum_exp_axis_1, CVXOPT),
    (kl_div, CVXOPT),
]

atoms = [
    ([
        (abs, (2, 2), [[[-5, 2], [-3, 1]]],
            Constant([[5, 2], [3, 1]])),
        (lambda x: cumsum(x, axis=1), (2, 2), [[[-5, 2], [-3, 1]]],
            Constant([[-5, 2], [-8, 3]])),
        (lambda x: cumsum(x, axis=0), (2, 2), [[[-5, 2], [-3, 1]]],
            Constant([[-5, -3], [-3, -2]])),
        (diag, (2,), [[[-5, 2], [-3, 1]]], Constant([-5, 1])),
        (diag, (2, 2), [[-5, 1]], Constant([[-5, 0], [0, 1]])),
        (exp, (2, 2), [[[1, 0], [2, -1]]],
            Constant([[math.e, 1], [math.e**2, 1.0/math.e]])),
        (huber, (2, 2), [[[0.5, -1.5], [4, 0]]],
            Constant([[0.25, 2], [7, 0]])),
        (lambda x: huber(x, 2.5), (2, 2), [[[0.5, -1.5], [4, 0]]],
         Constant([[0.25, 2.25], [13.75, 0]])),
        (inv_pos, (2, 2), [[[1, 2], [3, 4]]],
            Constant([[1, 1.0/2], [1.0/3, 1.0/4]])),
        (lambda x: (x + Constant(0))**-1, (2, 2), [[[1, 2], [3, 4]]],
            Constant([[1, 1.0/2], [1.0/3, 1.0/4]])),
        (kl_div, tuple(), [math.e, 1], Constant([1])),
        (kl_div, tuple(), [math.e, math.e], Constant([0])),
        (kl_div, (2,), [[math.e, 1], 1], Constant([1, 0])),
        (lambda x: kron(np.matrix("1 2; 3 4"), x), (4, 4), [np.matrix("5 6; 7 8")],
            Constant(np.kron(np.matrix("1 2; 3 4").A, np.matrix("5 6; 7 8").A))),
        (lambda_max, tuple(), [[[2, 0], [0, 1]]], Constant([2])),
        (lambda_max, tuple(), [[[2, 0, 0], [0, 3, 0], [0, 0, 1]]], Constant([3])),

        (lambda_max, tuple(), [[[5, 7], [7, -3]]], Constant([9.06225775])),
        (lambda x: lambda_sum_largest(x, 2), tuple(), [[[1, 2, 3], [2, 4, 5], [3, 5, 6]]], Constant([11.51572947])),
        (log_sum_exp, tuple(), [[[5, 7], [0, -3]]], Constant([7.1277708268])),
        (log_sum_exp_axis_0, (1, 2), [[[5, 7, 1], [0, -3, 6]]], Constant([[7.12910890], [6.00259878]])),
        (log_sum_exp_axis_1, (3,), [[[5, 7, 1], [0, -3, 6]]], Constant([5.00671535, 7.0000454, 6.0067153])),
        (logistic, (2, 2),
         [
             [[math.log(5), math.log(7)],
              [0,           math.log(0.3)]]],
         Constant(
             [[math.log(6), math.log(8)],
              [math.log(2), math.log(1.3)]])),
        (matrix_frac, tuple(), [[1, 2, 3],
                               [[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]]], Constant([14])),
        (matrix_frac, tuple(), [[1, 2, 3],
                               [[67, 78, 90],
                                [78, 94, 108],
                                [90, 108, 127]]], Constant([0.46557377049180271])),
        (matrix_frac, tuple(), [[[1, 2, 3],
                                [4, 5, 6]],
                               [[67, 78, 90],
                                [78, 94, 108],
                                [90, 108, 127]]], Constant([0.768852459016])),
        (maximum, (2,), [[-5, 2], [-3, 1], 0, [-1, 2]], Constant([0, 2])),
        (maximum, (2, 2), [[[-5, 2], [-3, 1]], 0, [[5, 4], [-1, 2]]],
            Constant([[5, 4], [0, 2]])),
        (max, tuple(), [[[-5, 2], [-3, 1]]], Constant([2])),
        (max, tuple(), [[-5, -10]], Constant([-5])),
        (lambda x: max(x, axis=0, keepdims=True), (1, 2), [[[-5, 2], [-3, 1]]], Constant([[2], [1]])),
        (lambda x: max(x, axis=1), (2,), [[[-5, 2], [-3, 1]]], Constant([-3, 2])),
        (lambda x: norm(x, 2), tuple(), [v_np], Constant([3])),
        (lambda x: norm(x, "fro"), tuple(), [[[-1, 2], [3, -4]]],
            Constant([5.47722557])),
        (lambda x: norm(x, 1), tuple(), [v_np], Constant([5])),
        (lambda x: norm(x, 1), tuple(), [[[-1, 2], [3, -4]]],
            Constant([10])),
        (lambda x: norm(x, "inf"), tuple(), [v_np], Constant([2])),
        (lambda x: norm(x, "inf"), tuple(), [[[-1, 2], [3, -4]]],
            Constant([4])),
        (lambda x: norm(x, "nuc"), tuple(), [[[2, 0], [0, 1]]], Constant([3])),
        (lambda x: norm(x, "nuc"), tuple(), [[[3, 4, 5], [6, 7, 8], [9, 10, 11]]],
            Constant([23.173260452512931])),
        (lambda x: norm(x, "nuc"), tuple(), [[[3, 4, 5], [6, 7, 8]]],
            Constant([14.618376738088918])),
        (lambda x: sum_largest(abs(x), 3), tuple(), [[1, 2, 3, -4, -5]], Constant([5+4+3])),
        (lambda x: mixed_norm(x, 1, 1), tuple(), [[[1, 2], [3, 4], [5, 6]]],
            Constant([21])),
        (lambda x: mixed_norm(x, 1, 1), tuple(), [[[1, 2, 3], [4, 5, 6]]],
            Constant([21])),
        # (lambda x: mixed_norm(x, 2, 1), tuple(), [[[3, 1], [4, math.sqrt(3)]]],
        #     Constant([7])),
        (lambda x: mixed_norm(x, 1, 'inf'), tuple(), [[[1, 4], [5, 6]]],
            Constant([10])),

        (pnorm, tuple(), [[1, 2, 3]], Constant([3.7416573867739413])),
        (lambda x: pnorm(x, 1), tuple(), [[1.1, 2, -3]], Constant([6.1])),
        (lambda x: pnorm(x, 2), tuple(), [[1.1, 2, -3]], Constant([3.7696153649941531])),
        (lambda x: pnorm(x, 2, axis=0), (2,), [[[1, 2], [3, 4]]], Constant([math.sqrt(5), 5.]).T),
        (lambda x: pnorm(x, 2, axis=1), (2,), [[[1, 2], [4, 5]]], Constant([math.sqrt(17), math.sqrt(29)])),
        (lambda x: pnorm(x, 'inf'), tuple(), [[1.1, 2, -3]], Constant([3])),
        (lambda x: pnorm(x, 3), tuple(), [[1.1, 2, -3]], Constant([3.3120161866074733])),
        (lambda x: pnorm(x, 5.6), tuple(), [[1.1, 2, -3]], Constant([3.0548953718931089])),
        (lambda x: pnorm(x, 1.2), tuple(), [[[1, 2, 3], [4, 5, 6]]], Constant([15.971021676279573])),

        (pos, tuple(), [8], Constant([8])),
        (pos, (2,), [[-3, 2]], Constant([0, 2])),
        (neg, (2,), [[-3, 3]], Constant([3, 0])),


        (lambda x: power(x, 1), tuple(), [7.45], Constant([7.45])),
        (lambda x: power(x, 2), tuple(), [7.45], Constant([55.502500000000005])),
        (lambda x: power(x, -1), tuple(), [7.45], Constant([0.1342281879194631])),
        (lambda x: power(x, -.7), tuple(), [7.45], Constant([0.24518314363015764])),
        (lambda x: power(x, -1.34), tuple(), [7.45], Constant([0.06781263100321579])),
        (lambda x: power(x, 1.34), tuple(), [7.45], Constant([14.746515290825071])),

        (quad_over_lin, tuple(), [[[-1, 2, -2], [-1, 2, -2]], 2], Constant([2*4.5])),
        (quad_over_lin, tuple(), [v_np, 2], Constant([4.5])),
        (lambda x: norm(x, 2), tuple(), [[[2, 0], [0, 1]]], Constant([2])),
        (lambda x: norm(x, 2), tuple(), [[[3, 4, 5], [6, 7, 8], [9, 10, 11]]], Constant([22.368559552680377])),
        (lambda x: scalene(x, 2, 3), (2, 2), [[[-5, 2], [-3, 1]]], Constant([[15, 4], [9, 2]])),
        (square, (2, 2), [[[-5, 2], [-3, 1]]], Constant([[25, 4], [9, 1]])),
        (sum, tuple(), [[[-5, 2], [-3, 1]]], Constant(-5)),
        (lambda x: sum(x, axis=0), (2,), [[[-5, 2], [-3, 1]]], Constant([-3, -2])),
        (lambda x: sum(x, axis=1), (2,), [[[-5, 2], [-3, 1]]], Constant([-8, 3])),
        (lambda x: (x + Constant(0))**2, (2, 2), [[[-5, 2], [-3, 1]]], Constant([[25, 4], [9, 1]])),
        (lambda x: sum_largest(x, 3), tuple(), [[1, 2, 3, 4, 5]], Constant([5+4+3])),
        (lambda x: sum_largest(x, 3), tuple(), [[[3, 4, 5], [6, 7, 8], [9, 10, 11]]], Constant([9+10+11])),
        (sum_squares, tuple(), [[[-1, 2], [3, -4]]], Constant([30])),
        (trace, tuple(), [[[3, 4, 5], [6, 7, 8], [9, 10, 11]]], Constant([3 + 7 + 11])),
        (trace, tuple(), [[[-5, 2], [-3, 1]]], Constant([-5 + 1])),
        (tv, tuple(), [[1, -1, 2]], Constant([5])),
        (tv, tuple(), [[1, -1, 2]], Constant([5])),
        (tv, tuple(), [[[-5, 2], [-3, 1]]], Constant([math.sqrt(53)])),
        (tv, tuple(), [[[-5, 2], [-3, 1]], [[6, 5], [-4, 3]], [[8, 0], [15, 9]]],
            Constant([LA.norm([7, -1, -8, 2, -10, 7])])),
        (tv, tuple(), [[[3, 4, 5], [6, 7, 8], [9, 10, 11]]], Constant([4*math.sqrt(10)])),
        (upper_tri, (3,), [[[3, 4, 5], [6, 7, 8], [9, 10, 11]]], Constant([6, 9, 10])),
        # # Advanced indexing.
        (lambda x: x[[1, 2], [0, 2]], (2,), [[[3, 4, 5], [6, 7, 8], [9, 10, 11]]], Constant([4, 11])),
        (lambda x: x[[1, 2]], (2, 2), [[[3, 4, 5], [6, 7, 8]]], Constant([[4, 5], [7, 8]])),
        (lambda x: x[np.array([[3, 4, 5], [6, 7, 8]]).T % 2 == 0], (2,), [[[3, 4, 5], [6, 7, 8]]],
         Constant([6, 4, 8])),
        (lambda x: x[2:0:-1], (2,), [[3, 4, 5]], Constant([5, 4])),
        (lambda x: x[2::-1], (3,), [[3, 4, 5]], Constant([5, 4, 3])),
        (lambda x: x[3:0:-1], (2,), [[3, 4, 5]], Constant([5, 4])),
        (lambda x: x[3::-1], (3,), [[3, 4, 5]], Constant([5, 4, 3])),
    ], Minimize),
    ([
        (entr, (2, 2), [[[1, math.e], [math.e**2, 1.0/math.e]]],
         Constant([[0, -math.e], [-2*math.e**2, 1.0/math.e]])),
        #(entr(0), Constant([0])),
        (log_det, tuple(),
         [[[20, 8, 5, 2],
           [8, 16, 2, 4],
           [5, 2, 5, 2],
           [2, 4, 2, 4]]], Constant([7.7424020218157814])),
        (geo_mean, tuple(), [[4, 1]], Constant([2])),
        (geo_mean, tuple(), [[0.01, 7]], Constant([0.2645751311064591])),
        (geo_mean, tuple(), [[63, 7]], Constant([21])),
        (geo_mean, tuple(), [[1, 10]], Constant([math.sqrt(10)])),
        (lambda x: geo_mean(x, [1, 1]), tuple(), [[1, 10]], Constant([math.sqrt(10)])),
        (lambda x: geo_mean(x, [.4, .8, 4.9]), tuple(), [[.5, 1.8, 17]], Constant([10.04921378316062])),

        (harmonic_mean, tuple(), [[1, 2, 3]], Constant([1.6363636363636365])),
        (harmonic_mean, tuple(), [[2.5, 2.5, 2.5, 2.5]], Constant([2.5])),
        (harmonic_mean, tuple(), [[0, 1, 2]], Constant([0])),

        (lambda x: diff(x, 0), (3,), [[1, 2, 3]], Constant([1, 2, 3])),
        (diff, (2,), [[1, 2, 3]], Constant([1, 1])),
        (diff, tuple(), [[1.1, 2.3]], Constant([1.2])),
        (lambda x: diff(x, 2), tuple(), [[1, 2, 3]], Constant([0])),
        (diff, (3,), [[2.1, 1, 4.5, -.1]], Constant([-1.1, 3.5, -4.6])),
        (lambda x: diff(x, 2), (2,), [[2.1, 1, 4.5, -.1]], Constant([4.6, -8.1])),
        (lambda x: diff(x, 1, axis=0), (1, 2), [[[-5, -3], [2, 1]]],
            Constant([[7], [4]])),
        (lambda x: diff(x, 1, axis=1), (2, 1), [[[-5, -3], [2, 1]]],
            Constant([[2, -1]])),

        (lambda x: pnorm(x, .5), tuple(), [[1.1, 2, .1]], Constant([7.724231543909264])),
        (lambda x: pnorm(x, -.4), tuple(), [[1.1, 2, .1]], Constant([0.02713620334])),
        (lambda x: pnorm(x, -1), tuple(), [[1.1, 2, .1]], Constant([0.0876494023904])),
        (lambda x: pnorm(x, -2.3), tuple(), [[1.1, 2, .1]], Constant([0.099781528576])),

        (lambda_min, tuple(), [[[2, 0], [0, 1]]], Constant([1])),
        (lambda_min, tuple(), [[[5, 7], [7, -3]]], Constant([-7.06225775])),
        (lambda x: lambda_sum_smallest(x, 2), tuple(), [[[1, 2, 3], [2, 4, 5], [3, 5, 6]]], Constant([-0.34481428])),
        (log, (2, 2), [[[1, math.e], [math.e**2, 1.0/math.e]]], Constant([[0, 1], [2, -1]])),
        (log1p, (2, 2), [[[0, math.e-1], [math.e**2-1, 1.0/math.e-1]]], Constant([[0, 1], [2, -1]])),
        (minimum, (2,), [[-5, 2], [-3, 1], 0, [1, 2]], Constant([-5, 0])),
        (minimum, (2, 2), [[[-5, 2], [-3, -1]], 0, [[5, 4], [-1, 2]]], Constant([[-5, 0], [-3, -1]])),
        (min, tuple(), [[[-5, 2], [-3, 1]]], Constant([-5])),
        (min, tuple(), [[-5, -10]], Constant([-10])),
        (lambda x: x**0.25, tuple(), [7.45], Constant([7.45**0.25])),
        (lambda x: x**0.32, (2,), [[7.45, 3.9]], Constant(np.power(np.array([7.45, 3.9]), 0.32))),
        (lambda x: x**0.9, (2, 2),  [[[7.45, 2.2], [4, 7]]], Constant(np.power(np.array([[7.45, 2.2], [4, 7]]).T, 0.9))),
        (sqrt, (2, 2), [[[2, 4], [16, 1]]], Constant([[1.414213562373095, 2], [4, 1]])),
        (lambda x: sum_smallest(x, 3), tuple(), [[-1, 2, 3, 4, 5]], Constant([-1+2+3])),
        (lambda x: sum_smallest(x, 4), tuple(), [[[-3, -4, 5], [6, 7, 8], [9, 10, 11]]], Constant([-3-4+5+6])),
        (lambda x: (x + Constant(0))**0.5, (2, 2), [[[2, 4], [16, 1]]], Constant([[1.414213562373095, 2], [4, 1]])),
    ],
        Maximize),
]


def check_solver(prob, solver_name):
    """Can the solver solve the problem?
    """
    try:
        if solver_name == ROBUST_CVXOPT:
            solver_name = CVXOPT
        chain = construct_solving_chain(prob, solver=solver_name)
        return True
    except SolverError:
        return False
    except:
        raise

# Tests numeric version of atoms.
def run_atom(atom, problem, obj_val, solver, verbose=False):
    assert problem.is_dcp()
    print(problem)
    if verbose:
        print(problem.objective)
        print(problem.constraints)
        print("solver", solver)
    if check_solver(problem, solver) and not (atom, solver) in KNOWN_SOLVER_ERRORS:
        tolerance = SOLVER_TO_TOL[solver]

        try:
            if solver == ROBUST_CVXOPT:
                result = problem.solve(solver=CVXOPT, verbose=verbose,
                                       kktsolver=ROBUST_KKTSOLVER)
            else:
                result = problem.solve(solver=solver, verbose=verbose)
        except SolverError as e:
            if (atom, solver) in KNOWN_SOLVER_ERRORS:
                return
            raise e

        if verbose:
            print(result)
            print(obj_val)
        assert(-tolerance <= (result - obj_val)/(1+np.abs(obj_val)) <= tolerance)


def get_indices(size):
    """Get indices for dimension.
    """
    if len(size) == 0:
        return tuple()
    elif len(size) == 1:
        return range(size[0])
    else:
        return itertools.product(range(size[0]), range(size[1]))

def test_constant_atoms():
    tests = []
    for atom_list, objective_type in atoms:
        for atom, size, args, obj_val in atom_list:
            for indexer in get_indices(size):
                for solver in SOLVERS_TO_TRY:
                    # Atoms with Constant arguments.
                    prob_val = obj_val[indexer].value
                    const_args = [Constant(arg) for arg in args]
                    problem = Problem(
                        objective_type(atom(*const_args)[indexer]))
                    yield (run_atom,
                           atom,
                           problem,
                           prob_val,
                           solver)
                    # Atoms with Variable arguments.
                    variables = []
                    constraints = []
                    for idx, expr in enumerate(args):
                        variables.append(Variable(intf.shape(expr)))
                        constraints.append(variables[-1] == expr)
                    objective = objective_type(atom(*variables)[indexer])
                    new_obj_val = prob_val
                    if objective_type == Maximize:
                        objective = -objective
                        new_obj_val = -new_obj_val
                    problem = Problem(objective, constraints)
                    yield (run_atom,
                           atom,
                           problem,
                           new_obj_val,
                           solver)
                    # Atoms with Parameter arguments.
                    parameters = []
                    for expr in args:
                        parameters.append(Parameter(intf.shape(expr)))
                        parameters[-1].value = intf.DEFAULT_INTF.const_to_matrix(expr)
                    objective = objective_type(atom(*parameters)[indexer])
                    yield (run_atom,
                           atom,
                           Problem(objective),
                           prob_val,
                           solver)
