"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from weakref import ref
import numpy as np
import cvxpy as cp
from cvxpy.constraints import constraint
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.perspective.perspective_utils import form_cone_constraint, form_perspective_from_f_exp
from cvxpy.constraints.exponential import ExpCone
from cvxpy.atoms.perspective import perspective
import pytest

@pytest.fixture
def quad_example():
    # Reference expected output
    x = cp.Variable()
    s = cp.Variable()
    obj = cp.quad_over_lin(x,s) + 2*x - 4*s
    constraints = [x >= 2,s <=.5 ]
    prob_ref = cp.Problem(cp.Minimize(obj),constraints)
    prob_ref.solve()

    return prob_ref.value, s.value, x.value

def test_form_perspective(quad_example):
    # form objective, introduce original variable
    x = cp.Variable()
    f_exp = cp.square(x)+2*x-4

    persp = form_perspective_from_f_exp(f_exp)
    constraints = persp.constraints + [persp.s <= .5, x>=2]
    prob = cp.Problem(cp.Minimize(persp.t),constraints)
    prob.solve()

    ref_val, ref_s, ref_x = quad_example

    assert np.isclose(prob.value,ref_val)
    assert np.isclose(x.value,ref_x)
    assert np.isclose(persp.s.value,ref_s)

@pytest.mark.parametrize("p",[1])
def test_p_norms(p):
    x = cp.Variable(3)
    f_exp = cp.norm(x,p)
    persp = form_perspective_from_f_exp(f_exp)
    constraints = persp.constraints + [persp.s <= 1, x >= [1,2,3]]
    prob = cp.Problem(cp.Minimize(persp.t),constraints)
    prob.solve()

    # reference problem
    ref_x = cp.Variable(3)
    ref_s = cp.Variable()

    obj = cp.power(cp.norm(ref_x,p),p) / cp.power(ref_s,p-1) 

    ref_constraints = [ref_x >= [1,2,3], ref_s <= 1]
    ref_prob = cp.Problem(cp.Minimize(obj),ref_constraints)
    ref_prob.solve()

    assert np.isclose(prob.value,ref_prob.value)
    assert np.allclose(x.value,ref_x.value)
    if p != 1: # s not used when denominator is s^0
        assert np.isclose(persp.s.value,ref_s.value) 

def test_rel_entr():
    x = cp.Variable()
    f_exp = -cp.log(x)

    persp = form_perspective_from_f_exp(f_exp)
    constraints = persp.constraints +[1 <=persp.s, 
                                    persp.s <= 2, 
                                    1 <= x,
                                    x <= 2]
    prob = cp.Problem(cp.Minimize(persp.t),constraints)
    prob.solve(solver=cp.MOSEK)

    # reference problem
    ref_x = cp.Variable()
    ref_s = cp.Variable() 
    obj = cp.rel_entr(ref_s,ref_x)

    ref_constraints = [1 <= ref_x, ref_x <= 2, 1 <= ref_s, ref_s <= 2]
    ref_prob = cp.Problem(cp.Minimize(obj),ref_constraints)
    ref_prob.solve(solver=cp.MOSEK)

    assert np.isclose(prob.value,ref_prob.value)
    assert np.allclose(x.value,ref_x.value)
    assert np.isclose(persp.s.value,ref_s.value) 

def test_exp():
    x = cp.Variable()
    f_exp = cp.exp(x)

    persp = form_perspective_from_f_exp(f_exp)
    constraints = persp.constraints + [persp.s >= 1, 1 <= x]
    prob = cp.Problem(cp.Minimize(persp.t),constraints)
    prob.solve(solver=cp.MOSEK)

    # reference problem
    ref_x = cp.Variable()
    ref_s = cp.Variable()
    ref_z = cp.Variable()

    obj = ref_z
    ref_constraints = [
        ExpCone(ref_x,ref_s,ref_z),
        ref_x >= 1, ref_s >= 1]
    ref_prob = cp.Problem(cp.Minimize(obj),ref_constraints)
    ref_prob.solve(solver=cp.MOSEK)

    assert np.isclose(prob.value,ref_prob.value)
    assert np.isclose(x.value,ref_x.value)
    assert np.isclose(persp.s.value,ref_s.value) 

def test_lse():
    x = cp.Variable(3)
    f_exp = cp.log_sum_exp(x)

    persp = form_perspective_from_f_exp(f_exp)
    constraints = persp.constraints + [1 <= persp.s, persp.s <= 2, [1,2,3] <= x]
    prob = cp.Problem(cp.Minimize(persp.t),constraints)
    prob.solve(solver=cp.MOSEK)

    # reference problem
    ref_x = cp.Variable(3)
    ref_s = cp.Variable()
    ref_z = cp.Variable(3)
    ref_t = cp.Variable()

    obj = ref_z
    ref_constraints = [
        ref_s >= cp.sum(ref_z),
        [1,2,3] <= ref_x, 1 <= ref_s, ref_s <= 2]
    ref_constraints += [ExpCone(ref_x[i]-ref_t,ref_s,ref_z[i]) for i in range(3)]
    ref_prob = cp.Problem(cp.Minimize(ref_t),ref_constraints)
    ref_prob.solve(solver=cp.MOSEK)

    assert np.isclose(prob.value,ref_prob.value)
    assert np.allclose(x.value,ref_x.value)
    assert np.isclose(persp.s.value,ref_s.value) 

def test_evaluate_persp():
    x = cp.Variable()
    s = cp.Variable(nonneg=True)
    f_exp = cp.square(x)+2*x-4
    obj = perspective(f_exp,s)

    l = np.array([2,2])
    val = obj.numeric(l) 
    assert np.isclose(val,2)

def test_atom(quad_example):
    # form objective, introduce original variable
    x = cp.Variable()
    s = cp.Variable(nonneg=True)
    
    f_exp = cp.square(x)+2*x-4

    obj = perspective(f_exp,s)
    # obj = -cp.log(x)

    constraints = [s <= .5, x>=2]
    prob = cp.Problem(cp.Minimize(obj),constraints)
    prob.solve()
    
    ref_val, ref_s, ref_x = quad_example

    assert np.isclose(prob.value,ref_val)
    assert np.isclose(x.value,ref_x)
    assert np.isclose(s.value,ref_s)
    
# More tests!