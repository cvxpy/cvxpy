import cvxpy as cp


class TestExpr2Smooth():
    def test_smooth(self):
        x = cp.Variable()
        y = cp.square(x)
        prob = cp.Problem(cp.Minimize(y), [x >= 0])
        prob.solve()
        assert prob.status == cp.OPTIMAL
