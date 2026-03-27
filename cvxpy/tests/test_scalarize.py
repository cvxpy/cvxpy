import pytest

import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
from cvxpy.transforms import scalarize


class ScalarizeTest(BaseTest):
    """
    Tests for the scalarize transform.
    """

    def setUp(self) -> None:
        
        self.x = cp.Variable()
        obj1 = cp.Minimize(cp.square(self.x))
        obj2 = cp.Minimize(cp.square(self.x-1))
        self.objectives = [obj1, obj2]

    def test_weighted_sum(self) -> None:
        """
        Test weighted sum.
        """
        weights = [1, 1]
        scalarized = scalarize.weighted_sum(self.objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 0.5, places=3)

        weights = [1, 0]
        scalarized = scalarize.weighted_sum(self.objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 0, places=3)

        weights = [0, 1]
        scalarized = scalarize.weighted_sum(self.objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 1, places=3)

    def test_targets_and_priorities(self) -> None:

        targets = [1, 1]
        priorities = [1, 1]
        scalarized = scalarize.targets_and_priorities(self.objectives, priorities, targets)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 0.5, places=3)

        targets = [1, 0]
        priorities = [1, 1]
        scalarized = scalarize.targets_and_priorities(self.objectives, priorities, targets)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 1, places=3)

        limits = [1, 0.25]
        targets = [0, 0]
        priorities = [1, 1e-4]
        scalarized = scalarize.targets_and_priorities(self.objectives, priorities, targets, limits,
                                                      off_target=1e-5)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 0.5, places=3)

        targets = [-1, 0]
        priorities = [1, 1]
        max_objectives = [cp.Maximize(-obj.args[0]) for obj in self.objectives]
        scalarized = scalarize.targets_and_priorities(max_objectives, priorities, targets,
                                                      off_target=1e-5)
        assert scalarized.args[0].is_concave()
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 1, places=3)

        limits = [-1, -0.25]
        targets = [0, 0]
        priorities = [1, 1e-4]
        max_objectives = [cp.Maximize(-obj.args[0]) for obj in self.objectives]
        scalarized = scalarize.targets_and_priorities(max_objectives, priorities, targets, limits,
                                                      off_target=1e-5)
        assert scalarized.args[0].is_concave()
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 0.5, places=3)


    def test_negative_priority_regression(self) -> None:
        # Regression: before the fix, delta and the indicator still used the
        # original targets[i]/limits[i] indices instead of the locally-flipped
        # tar/lim variables.  For priorities[i] < 0 this produced an indicator
        # constraint like (x-1)^2 <= -0.5 (never satisfiable), making the
        # problem INFEASIBLE.  With the fix the flipped values are used correctly
        # and the optimal x is 0.5.
        obj_2 = cp.Maximize(-self.objectives[1].args[0])
        objectives = [self.objectives[0], obj_2]
        priorities = [1, -1]
        targets = [1, -1]
        limits = [0.5, -0.5]
        off_target = 1e-2

        scalarized = scalarize.targets_and_priorities(
            objectives, priorities, targets, limits, off_target=off_target
        )
        prob = cp.Problem(scalarized)
        prob.solve(solver=cp.CLARABEL)

        self.assertEqual(prob.status, cp.OPTIMAL)
        self.assertAlmostEqual(float(self.x.value), 0.5, places=3)

    def test_mixed_convexity(self) -> None:
        obj_1 = self.objectives[0]
        obj_2 = cp.Maximize(-self.objectives[1].args[0])
        objectives = [obj_1, obj_2]
        targets = [1, -1]
        priorities = [1, 1]

        with pytest.raises(ValueError, match="Scalarized objective is neither convex nor concave"):
            scalarize.targets_and_priorities(objectives, priorities, targets)

        priorities = [1, -1]
        limits = [1, -1]
        scalarized = scalarize.targets_and_priorities(objectives, priorities, targets, limits)
        assert scalarized.args[0].is_convex()

        priorities = [-1, 1]
        limits = [1, -1]
        scalarized = scalarize.targets_and_priorities(objectives, priorities, targets, limits)
        assert scalarized.args[0].is_concave()


    def test_targets_and_priorities_exceptions(self) -> None:
        targets = [1, 1]

        # Test exceptions:
        priorities = [1]
        with pytest.raises(AssertionError, match="Number of objectives and priorities"):
            scalarize.targets_and_priorities(self.objectives, priorities, targets)

        priorities = [1, 1]
        targets = [1]
        with pytest.raises(AssertionError, match="Number of objectives and targets"):
            scalarize.targets_and_priorities(self.objectives, priorities, targets)

        priorities = [1, 1]
        targets = [1, 1]
        limits = [1]
        with pytest.raises(AssertionError, match="Number of objectives and limits"):
            scalarize.targets_and_priorities(self.objectives, priorities, targets, limits)

        limits = [1, 1]
        off_target = -1
        with pytest.raises(AssertionError, match="The off_target argument must be nonnegative"):
            scalarize.targets_and_priorities(self.objectives, priorities, targets, limits,
                                             off_target)


    def test_maximize_affine_targets_and_priorities(self) -> None:
        """Maximize(affine) must not be silently flipped to Minimize."""
        x = cp.Variable()

        # Single Maximize(x) with target=5, x in [0, 10]
        obj = cp.Maximize(x)
        scalarized = scalarize.targets_and_priorities(
            [obj], [1], [5], off_target=0.01
        )
        assert isinstance(scalarized, cp.Maximize)
        prob = cp.Problem(scalarized, [x <= 10, x >= 0])
        prob.solve(solver=cp.CLARABEL)
        assert prob.status == cp.OPTIMAL
        assert float(x.value) > 5.0, (
            f"Maximize(x) pushed x={x.value} below target 5"
        )

    def test_minimize_affine_negative_priority(self) -> None:
        """Minimize(affine) with negative priority should flip correctly."""
        x = cp.Variable()

        # Minimize(x) with priority=-1 flips to Maximize(-x)
        obj = cp.Minimize(x)
        scalarized = scalarize.targets_and_priorities(
            [obj], [-1], [5], off_target=0.01
        )
        assert isinstance(scalarized, cp.Maximize)
        prob = cp.Problem(scalarized, [x <= 10, x >= 0])
        prob.solve(solver=cp.CLARABEL)
        assert prob.status == cp.OPTIMAL
        assert float(x.value) < 5.0, (
            f"Negated Minimize(x) pushed x={x.value} above target"
        )

    def test_maximize_priority_leq_off_target(self) -> None:
        """Maximize objectives with priority <= off_target must not flip to Minimize.

        Regression test: when priority <= off_target, the pos(delta) coefficient
        vanishes or flips sign, making the expression affine or convex. The old
        code's is_convex() check would then wrap it in Minimize, silently
        reversing the optimization direction.
        """
        x = cp.Variable()

        # priority == off_target: expression is affine (pos term vanishes)
        scalarized = scalarize.targets_and_priorities(
            [cp.Maximize(x)], [1e-5], [3], off_target=1e-5
        )
        assert isinstance(scalarized, cp.Maximize), (
            "Maximize with priority==off_target should return Maximize"
        )
        prob = cp.Problem(scalarized, [x >= 0, x <= 10])
        prob.solve(solver=cp.CLARABEL)
        assert float(x.value) > 5.0, (
            f"Maximize(x) with priority==off_target gave x={x.value}"
        )

        # priority < off_target: expression becomes convex → should raise
        with pytest.raises(ValueError, match="not concave"):
            scalarize.targets_and_priorities(
                [cp.Maximize(x)], [0], [3], off_target=1e-5
            )

        # Multiple Maximize objectives, all priority == off_target
        y = cp.Variable()
        scalarized = scalarize.targets_and_priorities(
            [cp.Maximize(x), cp.Maximize(y)],
            [1e-5, 1e-5], [3, 7], off_target=1e-5
        )
        assert isinstance(scalarized, cp.Maximize)
        prob = cp.Problem(scalarized, [x >= 0, x <= 10, y >= 0, y <= 10])
        prob.solve(solver=cp.CLARABEL)
        assert float(x.value) > 5.0
        assert float(y.value) > 5.0

    def test_minimize_priority_leq_off_target(self) -> None:
        """Minimize(affine) with priority < off_target must not flip to Maximize."""
        x = cp.Variable()

        # priority < off_target with Minimize(affine) → should raise
        with pytest.raises(ValueError, match="not convex"):
            scalarize.targets_and_priorities(
                [cp.Minimize(x)], [0], [3], off_target=1e-5
            )

        # priority == off_target with Minimize(affine) → should stay Minimize
        scalarized = scalarize.targets_and_priorities(
            [cp.Minimize(x)], [1e-5], [3], off_target=1e-5
        )
        assert isinstance(scalarized, cp.Minimize), (
            "Minimize with priority==off_target should return Minimize"
        )
        prob = cp.Problem(scalarized, [x >= 0, x <= 10])
        prob.solve(solver=cp.CLARABEL)
        assert float(x.value) < 5.0, (
            f"Minimize(x) with priority==off_target gave x={x.value}"
        )

    def test_max(self) -> None:

        weights = [1, 2]
        scalarized = scalarize.max(self.objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 0.5858, places=3)

        weights = [2, 1]
        scalarized = scalarize.max(self.objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 0.4142, places=3)

    
    def test_log_sum_exp(self) -> None:
        weights = [1, 2]
        scalarized = scalarize.log_sum_exp(self.objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 0.6354, places=3)

        weights = [2, 1]
        scalarized = scalarize.log_sum_exp(self.objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 0.3646, places=3)
