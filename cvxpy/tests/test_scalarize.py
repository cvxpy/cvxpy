import pytest

import cvxpy as cp
from cvxpy.error import DCPError
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

    def test_weighted_sum_maximize_and_negative_weights(self) -> None:
        """Maximize objectives and negative weight flipping."""
        x = cp.Variable()
        # Maximize objectives: weighted sum should return Maximize
        objs = [cp.Maximize(-cp.square(x)), cp.Maximize(-cp.square(x - 1))]
        scalarized = scalarize.weighted_sum(objs, [1, 1])
        assert isinstance(scalarized, cp.Maximize)
        prob = cp.Problem(scalarized)
        prob.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(float(x.value), 0.5, places=3)

        # Negative weight on Minimize flips to Maximize
        obj = cp.Minimize(cp.square(x))
        scalarized = scalarize.weighted_sum([obj], [-1])
        assert isinstance(scalarized, cp.Maximize)

        # Mixed types via negative weight → DCPError
        with pytest.raises(DCPError):
            scalarize.weighted_sum(self.objectives, [-1, 1])

    def test_targets_and_priorities_penalty_values_minimize(self) -> None:
        """Verify exact penalty values for Minimize against the formula.

        penalty = ot * obj                          when obj < tar
        penalty = (p - ot) * pos(obj - tar) + ot * obj  when obj >= tar
        """
        x = cp.Variable()
        p, ot, tar = 2.0, 0.1, 1.0
        obj = cp.Minimize(cp.square(x))
        scalarized = scalarize.targets_and_priorities(
            [obj], [p], [tar], off_target=ot
        )

        # x=0, obj=0 < tar → penalty = 0.1 * 0 = 0
        prob = cp.Problem(scalarized, [x == 0])
        prob.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(prob.value, 0.0, places=4)

        # x=0.5, obj=0.25 < tar → penalty = 0.1 * 0.25 = 0.025
        prob = cp.Problem(scalarized, [x == 0.5])
        prob.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(prob.value, 0.025, places=4)

        # x=2, obj=4 > tar → penalty = 1.9*pos(4-1) + 0.1*4 = 5.7 + 0.4 = 6.1
        prob = cp.Problem(scalarized, [x == 2])
        prob.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(prob.value, 6.1, places=3)

    def test_targets_and_priorities_penalty_values_maximize(self) -> None:
        """Verify exact penalty values for Maximize against the formula.

        sign=-1: penalty = ot*obj when obj > tar
                 penalty = -(p-ot)*pos(tar-obj) + ot*obj when obj <= tar
        """
        x = cp.Variable()
        p, ot, tar = 2.0, 0.1, -1.0
        obj = cp.Maximize(-cp.square(x))
        scalarized = scalarize.targets_and_priorities(
            [obj], [p], [tar], off_target=ot
        )
        assert isinstance(scalarized, cp.Maximize)

        # x=0, obj=0 > tar=-1 → penalty = 0.1 * 0 = 0
        prob = cp.Problem(scalarized, [x == 0])
        prob.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(prob.value, 0.0, places=4)

        # x=2, obj=-4 < tar=-1
        # delta = -1*(-4-(-1)) = 3, expr = -1*1.9*3 + 0.1*(-4) = -6.1
        prob = cp.Problem(scalarized, [x == 2])
        prob.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(prob.value, -6.1, places=3)

    def test_maximize_negative_priority_with_limits(self) -> None:
        """Most complex code path: Maximize + negative priority + limits.

        Maximize(-x^2) with priority=-1 → flips to Minimize(x^2),
        tar=-1 → 1, lim=-4 → 4. Verifies the triple-flip logic.
        """
        x = cp.Variable()
        obj = cp.Maximize(-cp.square(x))
        scalarized = scalarize.targets_and_priorities(
            [obj], [-1], [-1], limits=[-4], off_target=0.01
        )
        assert isinstance(scalarized, cp.Minimize)
        prob = cp.Problem(scalarized, [x >= -5, x <= 5])
        prob.solve(solver=cp.CLARABEL)
        # Flipped to Minimize(x^2), target=1, optimal x=0 (below target)
        self.assertAlmostEqual(float(x.value), 0.0, places=2)
        # Limit x^2<=4 respected
        assert abs(float(x.value)) <= 2.0 + 1e-3

    def test_limits_enforced_minimize_and_maximize(self) -> None:
        """Limits act as hard bounds: upper for Minimize, lower for Maximize."""
        x = cp.Variable()

        # Minimize(x) with limit=3: x <= 3 enforced
        obj = cp.Minimize(x)
        scalarized = scalarize.targets_and_priorities(
            [obj], [1], [0], limits=[3], off_target=0.01
        )
        prob = cp.Problem(scalarized, [x >= -1, x <= 10])
        prob.solve(solver=cp.CLARABEL)
        assert float(x.value) <= 3.0 + 1e-3
        self.assertAlmostEqual(float(x.value), -1.0, places=2)

        # Maximize(x) with limit=2: x >= 2 enforced
        obj = cp.Maximize(x)
        scalarized = scalarize.targets_and_priorities(
            [obj], [1], [8], limits=[2], off_target=0.01
        )
        prob = cp.Problem(scalarized, [x >= 0, x <= 10])
        prob.solve(solver=cp.CLARABEL)
        assert float(x.value) >= 2.0 - 1e-3
        self.assertAlmostEqual(float(x.value), 10.0, places=2)

        # Infeasible: Minimize(x) limit=-1 with x >= 0
        obj = cp.Minimize(x)
        scalarized = scalarize.targets_and_priorities(
            [obj], [1], [0], limits=[-1], off_target=0.01
        )
        prob = cp.Problem(scalarized, [x >= 0, x <= 10])
        prob.solve(solver=cp.CLARABEL)
        assert prob.status in {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE}

    def test_off_target_edge_cases(self) -> None:
        """Edge cases: off_target=0, off_target==priority, off_target>priority."""
        x = cp.Variable()

        # off_target=0: no gradient below target
        obj = cp.Minimize(cp.square(x))
        scalarized = scalarize.targets_and_priorities(
            [obj], [1], [10], off_target=0
        )
        prob = cp.Problem(scalarized, [x >= -5, x <= 5])
        prob.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(float(x.value), 0.0, places=2)

        # priority == off_target: pos() term vanishes, target irrelevant
        scalarized = scalarize.targets_and_priorities(
            [obj], [0.5], [10], off_target=0.5
        )
        prob = cp.Problem(scalarized)
        prob.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(float(x.value), 0.0, places=3)

        # off_target > priority with quadratic → ValueError (not DCP)
        with pytest.raises(ValueError, match="neither convex nor concave"):
            scalarize.targets_and_priorities([obj], [0.1], [1], off_target=1.0)

        # off_target > priority with affine → concave, returns Maximize
        obj_aff = cp.Minimize(x)
        scalarized = scalarize.targets_and_priorities(
            [obj_aff], [0.1], [1], off_target=1.0
        )
        assert isinstance(scalarized, cp.Maximize)

    def test_all_negative_priorities(self) -> None:
        """Negative priorities flip all objectives."""
        x = cp.Variable()
        objs = [cp.Minimize(cp.square(x)), cp.Minimize(cp.square(x - 1))]
        scalarized = scalarize.targets_and_priorities(
            objs, [-1, -1], [-1, -1], off_target=1e-5
        )
        assert isinstance(scalarized, cp.Maximize)
        prob = cp.Problem(scalarized)
        prob.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(float(x.value), 0.5, places=3)

    def test_log_sum_exp_gamma_convergence(self) -> None:
        """log_sum_exp converges to weighted_sum (gamma→0) and max (gamma→∞)."""
        ws = scalarize.weighted_sum(self.objectives, [1, 1])
        prob = cp.Problem(ws)
        prob.solve(solver=cp.CLARABEL)
        x_ws = float(self.x.value)

        mx = scalarize.max(self.objectives, [1, 1])
        prob = cp.Problem(mx)
        prob.solve(solver=cp.CLARABEL)
        x_max = float(self.x.value)

        # Small gamma ≈ weighted_sum
        scalarized = scalarize.log_sum_exp(self.objectives, [1, 1], gamma=0.01)
        prob = cp.Problem(scalarized)
        prob.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(float(self.x.value), x_ws, places=2)

        # Large gamma ≈ max
        scalarized = scalarize.log_sum_exp(self.objectives, [1, 1], gamma=100)
        prob = cp.Problem(scalarized)
        prob.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(float(self.x.value), x_max, places=2)

    def test_max_and_log_sum_exp_maximize_not_dcp(self) -> None:
        """max() and log_sum_exp() with Maximize objectives are not DCP."""
        x = cp.Variable()
        objs = [cp.Maximize(-cp.square(x)), cp.Maximize(-cp.square(x - 1))]

        scalarized = scalarize.max(objs, [1, 1])
        assert not cp.Problem(scalarized).is_dcp()

        scalarized = scalarize.log_sum_exp(objs, [1, 1])
        assert not cp.Problem(scalarized).is_dcp()
