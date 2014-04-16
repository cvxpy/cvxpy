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
from cvxpy.atoms import *
from cvxpy.problems.objective import *
from cvxpy.problems.problem import Problem
from cvxpy.expressions.variables import Variable
from cvxpy.expressions.constants import Constant, Parameter
import cvxopt
import math
from nose.tools import assert_raises

TOL = 1e-3

v = cvxopt.matrix([-1,2,-2], tc='d')

atoms = [
    ([
        (abs([[-5,2],[-3,1]]), Constant([[5,2],[3,1]])),
        (exp([[1, 0],[2, -1]]), Constant([[math.e, 1],[math.e**2, 1.0/math.e]])),
        #(huber(0.5), 0.25),
        #(huber(-1.5), 2),
        (inv_pos([[1,2],[3,4]]), Constant([[1,1.0/2],[1.0/3,1.0/4]])),
        (kl_div(math.e, 1), Constant([1])),
        (kl_div(math.e, math.e), Constant([0])),
        (lambda_max([[2,0],[0,1]]), Constant([2])),
        (lambda_max([[5,7],[7,-3]]), Constant([9.06225775])),
        (log_sum_exp([[5, 7], [0, -3]]), Constant([7.1277708268])),
        (max([-5,2],[-3,1],0,[-1,2]), Constant([0,2])),
        (max([[-5,2],[-3,1]],0,[[5,4],[-1,2]]), Constant([[5,4],[0,2]])),
        #(norm(v), 3),
        (norm(v,2), Constant([3])),
        (norm([[-1, 2],[3, -4]], "fro"), Constant([5.47722557])),
        (norm(v,1), Constant([5])),
        (norm([[-1, 2], [3, -4]],1), Constant([10])),
        (norm(v,"inf"), Constant([2])),
        (norm([[-1, 2], [3, -4]],"inf"), Constant([4])),
        (norm([[2,0],[0,1]],"nuc"), Constant([3])),
        (norm([[3,4,5],[6,7,8],[9,10,11]],"nuc"), Constant([23.1733])),
        (pos(8), Constant([8])),
        (pos([-3,2]), Constant([0,2])),
        (neg([-3,3]), Constant([3,0])),
        #(pow_rat(4,1,1), 4),
        #(pow_rat(2,2,1), 4),
        #(pow_rat(4,2,2), 4),
        #(pow_rat(2,3,1), 8),
        #(pow_rat(4,3,2), 8),
        #(pow_rat(4,3,3), 4),
        #(pow_rat(2,4,1), 16),
        #(pow_rat(4,4,2), 16),
        #(pow_rat(8,4,3), 16),
        #(pow_rat(8,4,4), 8),
        (quad_over_lin(v, 2), Constant([4.5])),
        #(square_over_lin(2,4), 1),
        (norm([[2,0],[0,1]], 2), Constant([2])),
        (norm([[3,4,5],[6,7,8],[9,10,11]], 2), Constant([22.3686])),
        (square([[-5,2],[-3,1]]), Constant([[25,4],[9,1]])),
    ], Minimize),
    ([
        (entr([[1, math.e],[math.e**2, 1.0/math.e]]),
         Constant([[0, -math.e], [-2*math.e**2, 1.0/math.e]])),
        #(entr(0), Constant([0])),
        (log_det([[20, 8, 5, 2],
                  [8, 16, 2, 4],
                  [5, 2, 5, 2],
                  [2, 4, 2, 4]]), Constant([7.7424])),
        (geo_mean(4,1), Constant([2])),
        (geo_mean(2,2), Constant([2])),
        (lambda_min([[2,0],[0,1]]), Constant([1])),
        (lambda_min([[5,7],[7,-3]]), Constant([-7.06225775])),
        (log([[1, math.e],[math.e**2, 1.0/math.e]]), Constant([[0, 1],[2, -1]])),
        (min([-5,2],[-3,1],0,[1,2]), Constant([-5,0])),
        (min([[-5,2],[-3,-1]],0,[[5,4],[-1,2]]), Constant([[-5,0],[-3,-1]])),
        #(pow_rat(4,1,2), 2),
        #(pow_rat(8,1,3), 2),
        #(pow_rat(16,1,4),2),
        #(pow_rat(8,2,3), 4),
        #(pow_rat(4,2,4), 2),
        #(pow_rat(16,3,4),8),
        (sqrt([[2,4],[16,1]]), Constant([[1.414213562373095,2],[4,1]])),
    ], Maximize),
]

# Tests numeric version of atoms.
def run_atom(problem, obj_val):
    assert problem.is_dcp()
    print problem.objective
    print problem.constraints
    result = problem.solve()
    print result
    print obj_val
    assert( -TOL <= result - obj_val <= TOL )

def test_atom():
    for atom_list, objective_type in atoms:
        for atom, obj_val in atom_list:
            for row in xrange(atom.size[0]):
                for col in xrange(atom.size[1]):
                    # Atoms with Constant arguments.
                    yield run_atom, Problem(objective_type(atom[row,col])), obj_val[row,col].value
                    # Atoms with Variable arguments.
                    variables = []
                    constraints = []
                    for expr in atom.subexpressions:
                        variables.append( Variable(*expr.size) )
                        constraints.append( variables[-1] == expr)
                    atom_func = atom.__class__
                    objective = objective_type(atom_func(*variables)[row,col])
                    yield run_atom, Problem(objective, constraints), obj_val[row,col].value
                    # Atoms with Parameter arguments.
                    parameters = []
                    for expr in atom.subexpressions:
                        parameters.append( Parameter(*expr.size) )
                        parameters[-1].value = expr.value
                    objective = objective_type(atom_func(*parameters)[row,col])
                    yield run_atom, Problem(objective), obj_val[row,col].value
