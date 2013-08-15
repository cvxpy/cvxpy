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
import cvxopt
from nose.tools import assert_raises

TOL = 1e-5

v = cvxopt.matrix([-1,2,-2], tc='d')

convex_list = [
    (abs([[-5,2],[-3,1]]), [5,2,3,1]),
    #(huber(0.5), 0.25),
    #(huber(-1.5), 2),
    #(inv_pos(5), 0.2),
    (max([-5,2],[-3,1],0,[-1,2]), [0,2]),
    (max([[-5,2],[-3,1]],0,[[5,4],[-1,2]]), [5,4,0,2]),
    (normInf(v), [2]),
    #(norm(v), 3),
    (norm2(v), [3]),
    (norm1(v), [5]),
    (pos(8), [8]),
    (pos(-3), [0]),
    #(neg(-3), 3),
    #(neg(3), 0),
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
    (quad_over_lin(v, 2), [4.5]),
    #(square_over_lin(2,4), 1),
    (square([[-5,2],[-3,1]]), [25,4,9,1])
]

concave_list = [
    (geo_mean(4,1), [2]),
    (geo_mean(2,2), [2]),
    (min([-5,2],[-3,1],0,[1,2]), [-5,0]),
    (min([[-5,2],[-3,1]],0,[[5,4],[-1,2]]), [-5,-3,-1,0]),
    #(pow_rat(4,1,2), 2),
    #(pow_rat(8,1,3), 2),
    #(pow_rat(16,1,4),2),
    #(pow_rat(8,2,3), 4),
    #(pow_rat(4,2,4), 2),
    #(pow_rat(16,3,4),8),
    (sqrt([[2,4],[16,1]]), [1.414213562373095,2,4,1])
]

def run_atom(problem, obj_val):
    assert problem.is_dcp()
    result = problem.solve()
    assert( abs(result - obj_val) <= TOL )

def test_atom():
    for obj, obj_val in convex_list:
        for counter, obj_index in enumerate(obj):
            yield run_atom, Problem(Minimize(obj_index)), obj_val[counter]

    for obj, obj_val in concave_list:
        for counter, obj_index in enumerate(obj):
            yield run_atom, Problem(Maximize(obj_index)), obj_val[counter]