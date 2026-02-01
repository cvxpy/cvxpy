import numpy as np
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestBestOf():

    def test_circle_packing_best_of_one(self):
        np.random.seed(0)
        rng = np.random.default_rng(5)
        n = 5
        radius = rng.uniform(1.0, 3.0, n)
        centers = cp.Variable((n, 2), name='c')
        constraints = []
        for i in range(n - 1):
            constraints += [cp.sum((centers[i, :] - centers[i+1:, :]) ** 2, axis=1) >=
                            (radius[i] + radius[i+1:]) ** 2]
        obj = cp.Minimize(cp.max(cp.norm_inf(centers, axis=1) + radius))
        prob = cp.Problem(obj, constraints)
        centers.sample_bounds = [-5.0, 5.0]  
        n_runs = 10
        prob.solve(nlp=True, verbose=True, derivative_test='none', best_of=n_runs)
        obj_best_of = obj.value
        best_centers = centers.value
        all_objs = prob.solver_stats.extra_stats['all_objs_from_best_of']

        assert len(all_objs) == n_runs
        manual_obj = np.max(np.linalg.norm(best_centers, ord=np.inf, axis=1) + radius)
        assert manual_obj == obj_best_of
        assert manual_obj == np.min(all_objs)

    def test_path_planning_best_of_two(self):
        # test that if sample bounds and the value of the variables are set,
        # then best_of still initializes randomly within the bounds
        np.random.seed(0)
        rng = np.random.default_rng(5)
        n = 5
        radius = rng.uniform(1.0, 3.0, n)
        centers = cp.Variable((n, 2), name='c')
        constraints = []
        for i in range(n - 1):
            constraints += [cp.sum((centers[i, :] - centers[i+1:, :]) ** 2, axis=1) >=
                            (radius[i] + radius[i+1:]) ** 2]
        obj = cp.Minimize(cp.max(cp.norm_inf(centers, axis=1) + radius))
        prob = cp.Problem(obj, constraints)
        
        centers.value = np.random.rand(n, 2)
        centers.sample_bounds = [-5.0, 5.0]  
        n_runs = 10
        prob.solve(nlp=True, verbose=True, derivative_test='none', best_of=n_runs)
        obj_best_of = obj.value
        best_centers = centers.value
        all_objs = prob.solver_stats.extra_stats['all_objs_from_best_of']
        _, counts = np.unique(all_objs, return_counts=True)

        assert np.max(counts) == 1
        assert len(all_objs) == n_runs
        manual_obj = np.max(np.linalg.norm(best_centers, ord=np.inf, axis=1) + radius)
        assert manual_obj == obj_best_of
        assert manual_obj == np.min(all_objs)

    def test_path_planning_best_of_three(self):
        # test that no error is raised when best_of > 1 and all variables have finite bounds
        x = cp.Variable(bounds=[-5, 5])
        y = cp.Variable(bounds=[-3, 3])
        obj = cp.Minimize((x - 1) ** 2 + (y - 2) ** 2)
        prob = cp.Problem(obj)
        prob.solve(nlp=True, best_of=3)

        all_objs = prob.solver_stats.extra_stats['all_objs_from_best_of']
        assert len(all_objs) == 3

    def test_path_planning_best_of_four(self):
        # test that an error is raised it there is a variable with one 
        # infinite bound and no sample_bounds when best_of > 1
        x = cp.Variable(bounds=[-5, 5])
        y = cp.Variable(bounds=[-3, None])
        obj = cp.Minimize((x - 1) ** 2 + (y - 2) ** 2)
        prob = cp.Problem(obj)

        # test that it raises an error
        with pytest.raises(ValueError):
            prob.solve(nlp=True, best_of=3)

    def test_path_planning_best_of_five(self):
        # test that no error is raised it there is a variable with 
        # no bounds and no sample bounds, but it has been assigned 
        # a value
        x = cp.Variable(bounds=[-5, 5])
        y = cp.Variable()
        y.value = 5
        obj = cp.Minimize((x - 1) ** 2 + (y - 2) ** 2)
        prob = cp.Problem(obj)
        prob.solve(nlp=True, best_of=3)
        all_objs = prob.solver_stats.extra_stats['all_objs_from_best_of']
        assert len(all_objs) == 3