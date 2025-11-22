import pytest

import cvxpy as cp


class TestInterfaces:

    def test_knitro_call(self):
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize((x - 2) ** 2), [x >= 1])
        with pytest.raises(NotImplementedError):
            prob.solve(solver=cp.KNITRO, nlp=True)

    def test_copt_call(self):
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize((x - 2) ** 2), [x >= 1])
        with pytest.raises(NotImplementedError):
            prob.solve(solver=cp.COPT, nlp=True)
