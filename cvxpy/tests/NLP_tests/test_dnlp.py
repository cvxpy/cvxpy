import cvxpy as cp


class TestDNLP():
    """
    This class tests whether problems are correctly identified as DNLP
    (disciplined nonlinear programs) and whether the objective and constraints
    are correctly identified as smooth, esr or hsr.

    We adopt the convention that a function is smooth if and only if it is
    both esr and hsr. This convention is analogous to DCP and convex programming
    where a function is affine if and only if it is both convex and concave.
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

    def test_abs_esr(self):
        x = cp.Variable()
        objective = cp.Minimize(cp.abs(x))
        prob = cp.Problem(objective)
        assert objective.expr.is_esr()
        assert prob.is_dnlp()

    def test_sqrt_hsr(self):
        x = cp.Variable()
        objective = cp.Maximize(cp.sqrt(x))
        prob = cp.Problem(objective)
        assert objective.expr.is_hsr()
        assert prob.is_dnlp()

    def test_simple_neg_expr(self):
        x = cp.Variable()
        y = cp.Variable()
        constraints = [cp.abs(x) - cp.sqrt(y) <= 5]
        assert constraints[0].is_dnlp()
        assert constraints[0].expr.is_esr()

    def test_non_dnlp(self):
        """
        The constraint abs(x) >= 5 is hsr but makes 
        the problem non-DNLP.
        """
        x = cp.Variable()
        constraints = [cp.abs(x) >= 5]
        # the expression is 5 + -abs(x)
        assert constraints[0].expr.is_hsr()
        assert not constraints[0].is_dnlp()

    def test_simple_composition(self):
        x = cp.Variable()
        obj1 = cp.Minimize(cp.log(cp.abs(x)))
        assert obj1.is_dnlp()

        obj2 = cp.Minimize(cp.exp(cp.norm1(x)))
        assert obj2.is_dnlp()

        expr = cp.sqrt(cp.abs(x))
        assert not expr.is_dnlp()
    
    def test_complicated_composition(self):
        x = cp.Variable()
        y = cp.Variable()
        expr = cp.minimum(cp.sqrt(cp.exp(x)), -cp.abs(y))
        assert expr.is_hsr()

        # cannot minimize an hsr function
        obj = cp.Minimize(expr)
        prob = cp.Problem(obj)
        assert not prob.is_dnlp()
