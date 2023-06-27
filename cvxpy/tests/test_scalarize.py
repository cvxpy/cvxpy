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
        self.assertItemsAlmostEqual(self.x.value, 0.5)

        weights = [1, 0]
        scalarized = scalarize.weighted_sum(self.objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 0)

        weights = [0, 1]
        scalarized = scalarize.weighted_sum(self.objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 1)

    def test_targets_and_priorities(self) -> None:

        targets = [1, 1]
        priorities = [1, 1]
        scalarized = scalarize.targets_and_priorities(self.objectives, priorities, targets)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 0.5)

        targets = [1, 0]
        priorities = [1, 1]
        scalarized = scalarize.targets_and_priorities(self.objectives, priorities, targets)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 1, places=4)

        limits = [1, 0.25]
        targets = [0, 0]
        priorities = [1, 0]
        scalarized = scalarize.targets_and_priorities(self.objectives, priorities, targets, limits, 
                                                      off_target=0)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 0.5, places=4)

    def test_max(self) -> None:

        weights = [1, 2]
        scalarized = scalarize.max(self.objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 0.5858, places=4)

        weights = [2, 1]
        scalarized = scalarize.max(self.objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 0.4142, places=4)

    
    def test_log_sum_exp(self) -> None:
        weights = [1, 2]
        scalarized = scalarize.log_sum_exp(self.objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 0.6354, places=4)

        weights = [2, 1]
        scalarized = scalarize.log_sum_exp(self.objectives, weights)
        prob = cp.Problem(scalarized)
        prob.solve()
        self.assertItemsAlmostEqual(self.x.value, 0.3646, places=4)