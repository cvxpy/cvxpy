import numpy as np
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestScalarProblems():

    def test_exp(self):
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.exp(x)), [x >= 4])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_entropy(self):
        x = cp.Variable()
        prob = cp.Problem(cp.Maximize(cp.entr(x)), [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL
    
    # skip
    @pytest.mark.skip(reason="Skipping KL test for now")
    def test_KL(self):
       p = cp.Variable()
       q = cp.Variable()
       prob = cp.Problem(cp.Minimize(cp.kl_div(p, q)), [p >= 0.1, q >= 0.1, p + q == 2])
       prob.solve(nlp=True, solver=cp.IPOPT)
       assert prob.status == cp.OPTIMAL
    
    @pytest.mark.skip(reason="Skipping KL test for now")
    def test_KL_matrix(self):
        Y = cp.Variable((3, 3))
        X = cp.Variable((3, 3))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.kl_div(X, Y))), 
                        [X >= 0.1, Y >= 0.1, cp.sum(X) + cp.sum(Y) == 6])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL
        
    def test_entropy_matrix(self):
        x = cp.Variable((3, 2))
        prob = cp.Problem(cp.Maximize(cp.sum(cp.entr(x))), [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL
        
    def test_logistic(self):
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.logistic(x)), [x >= 0.4])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL
    
    def test_logistic_matrix(self):
        x = cp.Variable((3, 2))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.logistic(x))), [x >= 0.4])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_power(self):
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.power(x, 3)), [x >= 2])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_power_matrix(self):
        x = cp.Variable((3, 2))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.power(x, 3))), [x >= 2])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_power_fractional(self):
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.power(x, 1.5)), [x >= 2])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

        x = cp.Variable((3, 2))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.power(x, 0.6))), [x >= 2])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_power_fractional_matrix(self):
        x = cp.Variable((3, 2))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.power(x, 1.5))), [x >= 2])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

        x = cp.Variable((3, 2))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.power(x, 0.6))), [x >= 2])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL
    
    def test_scalar_trig(self):
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.tan(x)), [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

        prob = cp.Problem(cp.Minimize(cp.sin(x)), [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

        prob = cp.Problem(cp.Minimize(cp.cos(x)), [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_matrix_trig(self):
        x = cp.Variable((3, 2))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.tan(x))), [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

        prob = cp.Problem(cp.Minimize(cp.sum(cp.sin(x))), [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

        prob = cp.Problem(cp.Minimize(cp.sum(cp.cos(x))), [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_xexp(self):
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.xexp(x)), [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

        x = cp.Variable((3, 2))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.xexp(x))), [x >= 0.1])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_scalar_quad_form(self):
        x = cp.Variable((1, ))
        P = np.array([[3]])
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P)), [x >= 1])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL
    
    def test_scalar_quad_over_lin(self):
        x = cp.Variable()
        y = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.quad_over_lin(x, y)), [x >= 1, y <= 1])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL
    
    def test_matrix_quad_over_lin(self):
        x = cp.Variable((3, 2))
        y = cp.Variable((1, ))
        prob = cp.Problem(cp.Minimize(cp.quad_over_lin(x, y)), [x >= 1, y <= 1])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

        y = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.quad_over_lin(x, y)), [x >= 1, y <= 1])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL
        
    def test_rel_entr_both_scalar_variables(self):
        x = cp.Variable()
        y = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.rel_entr(x, y)),
                          [x >= 0.1, y >= 0.1, x <= 2, y <= 2])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

        x = cp.Variable((1, ))
        y = cp.Variable((1, ))
        prob = cp.Problem(cp.Minimize(cp.rel_entr(x, y)),
                          [x >= 0.1, y >= 0.1, x <= 2, y <= 2])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL
    
    def test_rel_entr_matrix_variable_and_scalar_variable(self):
        x = cp.Variable((3, 2))
        y = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.sum(cp.rel_entr(x, y))),
                          [x >= 0.1, y >= 0.1, x <= 2, y <= 2])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_rel_entr_scalar_variable_and_matrix_variable(self):
        x = cp.Variable()
        y = cp.Variable((3, 2))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.rel_entr(x, y))),
                          [x >= 0.1, y >= 0.1, x <= 2, y <= 2])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_rel_entr_both_matrix_variables(self):
        x = cp.Variable((3, 2))
        y = cp.Variable((3, 2))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.rel_entr(x, y))),
                          [x >= 0.1, y >= 0.1, x <= 2, y <= 2])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL

    def test_rel_entr_both_vector_variables(self):
        x = cp.Variable((3, ))
        y = cp.Variable((3, ))
        prob = cp.Problem(cp.Minimize(cp.sum(cp.rel_entr(x, y))),
                          [x >= 0.1, y >= 0.1, x <= 2, y <= 2])
        prob.solve(nlp=True, solver=cp.IPOPT)
        assert prob.status == cp.OPTIMAL
