import pytest

import cvxpy as cp
from cvxpy import error


class TestDNLP():
    """
    This class tests whether problems are correctly identified as DNLP
    (disciplined nonlinear programs) and whether the objective and constraints
    are correctly identified as linearizable, linearizable convex or linearizable concave.

    We adopt the convention that a function is smooth if and only if it is
    both linearizable convex and linearizable concave. This convention is analogous to DCP
    and convex programming where a function is affine iff it is both convex and concave.
    """

    def test_simple_smooth(self):
        # Define a simple nonlinear program
        x = cp.Variable()
        y = cp.Variable()
        objective = cp.Minimize(cp.log(x - 1) + cp.exp(y - 2))
        constraints = [x + y == 1]
        prob = cp.Problem(objective, constraints)

        assert prob.is_dnlp()
        assert prob.objective.expr.is_smooth()
        assert prob.constraints[0].expr.is_smooth()

    def test_abs_linearizable_convex(self):
        x = cp.Variable()
        objective = cp.Minimize(cp.abs(x))
        prob = cp.Problem(objective)
        assert objective.expr.is_linearizable_convex()
        assert prob.is_dnlp()

    def test_sqrt_linearizable_concave(self):
        x = cp.Variable()
        objective = cp.Maximize(cp.sqrt(x))
        prob = cp.Problem(objective)
        assert objective.expr.is_linearizable_concave()
        assert prob.is_dnlp()

    def test_simple_neg_expr(self):
        x = cp.Variable()
        y = cp.Variable()
        constraints = [cp.abs(x) - cp.sqrt(y) <= 5]
        assert constraints[0].is_dnlp()
        assert constraints[0].expr.is_linearizable_convex()

    def test_non_dnlp(self):
        """
        The constraint abs(x) >= 5 is smooth concave but makes 
        the problem non-DNLP.
        """
        x = cp.Variable()
        constraints = [cp.abs(x) >= 5]
        # the expression is 5 + -abs(x)
        assert constraints[0].expr.is_linearizable_concave()
        assert not constraints[0].is_dnlp()

    def test_simple_composition(self):
        x = cp.Variable()
        obj1 = cp.Minimize(cp.log(cp.abs(x)))
        assert obj1.is_dnlp()

        obj2 = cp.Minimize(cp.exp(cp.norm1(x)))
        assert obj2.is_dnlp()

        expr = cp.sqrt(cp.abs(x))
        # we treat sqrt as linearizable
        assert expr.is_dnlp()
    
    def test_complicated_composition(self):
        x = cp.Variable()
        y = cp.Variable()
        expr = cp.minimum(cp.sqrt(cp.exp(x)), -cp.abs(y))
        assert expr.is_linearizable_concave()

        # cannot minimize a linearizable concave function
        obj = cp.Minimize(expr)
        prob = cp.Problem(obj)
        assert not prob.is_dnlp()


class TestNonDNLP:
    
    def test_max(self):
        x = cp.Variable(1)
        y = cp.Variable(1)

        objective = cp.Maximize(cp.maximum(x, y))

        constraints = [x - 14 == 0, y - 6 == 0]

        # assert raises DNLP error
        problem = cp.Problem(objective, constraints)
        with pytest.raises(error.DNLPError):
            problem.solve(solver=cp.IPOPT, nlp=True)

    def test_min(self):
        x = cp.Variable(1)
        y = cp.Variable(1)

        objective = cp.Minimize(cp.minimum(x, y))

        constraints = [x - 14 == 0, y - 6 == 0]

        problem = cp.Problem(objective, constraints)
        with pytest.raises(error.DNLPError):
            problem.solve(solver=cp.IPOPT, nlp=True)

    def test_max_2(self):
        # Define variables
        x = cp.Variable(3)
        y = cp.Variable(3)

        objective = cp.Maximize(cp.sum(cp.maximum(x, y)))

        constraints = [x <= 14, y <= 14]

        problem = cp.Problem(objective, constraints)
        with pytest.raises(error.DNLPError):
            problem.solve(solver=cp.IPOPT, nlp=True)

