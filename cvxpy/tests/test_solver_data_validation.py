
"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import pytest
import scipy.sparse as sp

import cvxpy as cp
from cvxpy.tests.base_test import BaseTest


class TestSolverDataValidation(BaseTest):
    def test_inf_constant_in_constraint(self):
        x = cp.Variable()
        # Constant(np.inf) is allowed in expression tree
        c = cp.Constant(np.inf)
        prob = cp.Problem(cp.Minimize(x), [x >= c])
        
        # Should raise ValueError from our new check in solving_chain.py
        # We use SCS because it's a conic solver and usually available
        if cp.SCS in cp.installed_solvers():
            with pytest.raises(ValueError, match="Problem data in .* contains NaN or Inf"):
                prob.solve(solver=cp.SCS)

    def test_nan_constant_in_objective(self):
        x = cp.Variable()
        # Use NaN in the linear coefficient, which goes into data[s.C]
        prob = cp.Problem(cp.Minimize(x * cp.Constant(np.nan)), [x >= 0])
        
        if cp.SCS in cp.installed_solvers():
            with pytest.raises(ValueError, match="Problem data in .* contains NaN or Inf"):
                prob.solve(solver=cp.SCS)

    def test_ignore_nan_flag(self):
        x = cp.Variable()
        c = cp.Constant(np.inf)
        prob = cp.Problem(cp.Minimize(x), [x >= c])
        
        if cp.SCS in cp.installed_solvers():
            # Should NOT raise ValueError (might raise SolverError or return infeasible/unbounded)
            try:
                prob.solve(solver=cp.SCS, ignore_nan=True)
            except ValueError as e:
                if "Problem data in" in str(e):
                    pytest.fail("ValueError raised even with ignore_nan=True")
            except Exception:
                # Other errors are expected (SolverError, etc.)
                pass

    def test_sparse_matrix_check(self):
        # Construct a problem where A matrix might end up sparse and containing Inf
        # It's hard to force a sparse matrix with Inf through standard atoms without it being caught
        # elsewhere,
        # so we might need to manually invoke the check or mock the data.
        
        # Let's try to manually call solve_via_data on a SolvingChain to ensure we hit the sparse
        # path.
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= 1])
        
        # Create a dummy chain
        if cp.OSQP in cp.installed_solvers():
            solver = cp.OSQP
        elif cp.SCS in cp.installed_solvers():
            solver = cp.SCS
        else:
            return

        # We can manually trigger the check by creating a SolvingChain and calling solve_via_data
        # with bad data
        # But SolvingChain.solve_via_data calls self.solver.solve_via_data.
        # We need a real chain.
        
        chain = prob._construct_chain(solver=solver)
        
        # Create bad data
        bad_data = {
            cp.settings.A: sp.csc_matrix(np.array([[np.inf]])),
            cp.settings.B: np.array([1.0]),
            cp.settings.C: np.array([1.0]),
            cp.settings.OFFSET: 0.0
        }
        
        # We need to mock the solver instance inside the chain to avoid actual solving
        # or just expect the error before the solver is called.
        
        # The check happens BEFORE self.solver.solve_via_data
        
        with pytest.raises(ValueError, match="Problem data in .* contains NaN or Inf"):
            chain.solve_via_data(prob, bad_data)

    def test_solver_opts_none(self):
        # Test that solver_opts=None is handled correctly (defaults to {})
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= 1])
        
        if cp.SCS in cp.installed_solvers():
            solver = cp.SCS
        else:
            return

        chain = prob._construct_chain(solver=solver)
        
        # Good data
        good_data = {
            cp.settings.A: sp.csc_matrix(np.array([[1.0]])),
            cp.settings.B: np.array([1.0]),
            cp.settings.C: np.array([1.0]),
            cp.settings.OFFSET: 0.0
        }
        
        # Should not raise error
        try:
            # We mock the solver's solve_via_data to do nothing
            original_solve = chain.solver.solve_via_data
            chain.solver.solve_via_data = lambda *args, **kwargs: {}
            
            chain.solve_via_data(prob, good_data, solver_opts=None)
            
            # Restore
            chain.solver.solve_via_data = original_solve
        except Exception as e:
            pytest.fail(f"Raised exception with solver_opts=None: {e}")

