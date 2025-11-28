import numpy as np
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestBestOf():

    def test_circle_packing_best_of(self):
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