
import numpy as np

import cvxpy as cp


class TestProblem():
    """
    This class can be used to test internal function for Problem that have been added 
    in the DNLP extension.
    """

    def test_set_initial_point_both_bounds_infinity(self):
        # when both bounds are infinity, the initial point should be zero vector
                
        # test 1
        x = cp.Variable((3, ))
        prob = cp.Problem(cp.Minimize(cp.sum(x)))
        prob.set_NLP_initial_point()
        assert (x.value == np.zeros((3, ))).all()

        # test 2
        x = cp.Variable((3, ), bounds=[None, None])
        prob = cp.Problem(cp.Minimize(cp.sum(x)))
        prob.set_NLP_initial_point()
        assert (x.value == np.zeros((3, ))).all()

        # test 3
        x = cp.Variable((3, ), bounds=[-np.inf, np.inf])
        prob = cp.Problem(cp.Minimize(cp.sum(x)))
        prob.set_NLP_initial_point()
        assert (x.value == np.zeros((3, ))).all()

        # test 4
        x = cp.Variable((3, ), bounds=[None, np.inf])
        prob = cp.Problem(cp.Minimize(cp.sum(x)))
        prob.set_NLP_initial_point()
        assert (x.value == np.zeros((3, ))).all()

        # test 5
        x = cp.Variable((3, ), bounds=[-np.inf, None])
        prob = cp.Problem(cp.Minimize(cp.sum(x)))
        prob.set_NLP_initial_point()
        assert (x.value == np.zeros((3, ))).all()

        
    def test_set_initial_point_lower_bound_infinity(self):
        # when one bound is infinity, the initial point should be one unit
        # away from the finite bound

        # test 1
        x = cp.Variable((3, ), bounds=[None, 3.5])
        prob = cp.Problem(cp.Minimize(cp.sum(x)))
        prob.set_NLP_initial_point()
        assert (x.value == 2.5 * np.ones((3, ))).all()

        # test 2
        x = cp.Variable((3, ), bounds=[-np.inf, 3.5])
        prob = cp.Problem(cp.Minimize(cp.sum(x)))
        prob.set_NLP_initial_point()
        assert (x.value == 2.5 * np.ones((3, ))).all()
    
    def test_set_initial_point_upper_bound_infinity(self):
        # when one bound is infinity, the initial point should be one unit
        # away from the finite bound

        # test 1
        x = cp.Variable((3, ), bounds=[3.5, None])
        prob = cp.Problem(cp.Minimize(cp.sum(x)))
        prob.set_NLP_initial_point()
        assert (x.value == 4.5 * np.ones((3, ))).all()

        # test 2
        x = cp.Variable((3, ), bounds=[3.5, np.inf])
        prob = cp.Problem(cp.Minimize(cp.sum(x)))
        prob.set_NLP_initial_point()
        assert (x.value == 4.5 * np.ones((3, ))).all()

    def test_set_initial_point_both_bounds_finite(self):
        # when both bounds are finite, the initial point should be the midpoint
        # between the two bounds

        # test 1
        x = cp.Variable((3, ), bounds=[3.5, 4.5])
        prob = cp.Problem(cp.Minimize(cp.sum(x)))
        prob.set_NLP_initial_point()
        assert (x.value == 4.0 * np.ones((3, ))).all()

    def test_set_initial_point_mixed_inf_and_finite(self):
        lb = np.array([-np.inf, 3.5, -np.inf, -1.5, 2, 2.5])
        ub = np.array([-4, 4.5, np.inf, 4.5, np.inf, 4.5])
        x = cp.Variable((6, ), bounds=[lb, ub])
        prob = cp.Problem(cp.Minimize(cp.sum(x)))
        prob.set_NLP_initial_point()
        expected = np.array([-5, 4.0, 0.0, 1.5, 3, 3.5])
        assert (x.value == expected).all()

    def test_set_initial_point_two_variables(self):
        x = cp.Variable((2, ), bounds=[-np.inf, np.inf])
        y = cp.Variable((2, ), bounds=[-3, np.inf])
        prob = cp.Problem(cp.Minimize(cp.sum(x) + cp.sum(y)))
        prob.set_NLP_initial_point()
        assert (x.value == np.zeros((2, ))).all()
        assert (y.value == -2 * np.ones((2, ))).all()

    def test_set_initial_point_nonnegative_attributes(self):
        x = cp.Variable((2, ), nonneg=True)
        prob = cp.Problem(cp.Minimize(cp.sum(x)))
        prob.set_NLP_initial_point()
        assert (x.value == np.ones((2, ))).all()

    def test_set_initial_point_nonnegative_attributes_and_bounds(self):
        x = cp.Variable((2, ), nonneg=True, bounds=[1, None])
        prob = cp.Problem(cp.Minimize(cp.sum(x)))
        prob.set_NLP_initial_point()
        assert (x.value == 2 * np.ones((2, ))).all()