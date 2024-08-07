import numpy as np

import cvxpy as cp
from cvxpy.constraints.nonpos import Inequality
from cvxpy.constraints.psd import PSD
from cvxpy.constraints.zero import Equality
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr


class Test_Attributes():

    def test_multiple_attributes(self) -> None:
        x = cp.Variable(shape=(2,2), symmetric=True, nonneg=True, integer=True)
        target = np.array(np.eye(2) * 5)
        prob = cp.Problem(cp.Minimize(0), [x == target])
        new_prob = CvxAttr2Constr(prob, reduce_bounds=True)
        assert type(new_prob.apply(prob)[0].constraints[0]) is Inequality
        assert type(new_prob.apply(prob)[0].constraints[1]) is Equality

        prob.solve()
        assert np.allclose(x.value, target)

    def test_nonneg_PSD(self) -> None:
        x = cp.Variable(shape=(2,2), PSD=True, nonneg=True)
        target = np.array(np.eye(2) * 5)
        prob = cp.Problem(cp.Minimize(0), [x == target])
        new_prob = CvxAttr2Constr(prob, reduce_bounds=True)
        assert type(new_prob.apply(prob)[0].constraints[0]) is PSD
        assert type(new_prob.apply(prob)[0].constraints[1]) is Inequality
        assert type(new_prob.apply(prob)[0].constraints[2]) is Equality

        prob.solve()
        assert np.allclose(x.value, target)

    def test_nonpos_NSD(self) -> None:
        x = cp.Variable(shape=(2,2), NSD=True, nonpos=True)
        target = np.array(np.eye(2) * 5)
        prob = cp.Problem(cp.Minimize(0), [x == -target])

        new_prob = CvxAttr2Constr(prob, reduce_bounds=True)
        assert type(new_prob.apply(prob)[0].constraints[0]) is PSD
        assert type(new_prob.apply(prob)[0].constraints[1]) is Inequality
        assert type(new_prob.apply(prob)[0].constraints[2]) is Equality

        prob.solve()
        assert np.allclose(x.value, -target)

    def test_integer_bounds(self) -> None:
        x = cp.Variable(shape=(2,2), integer=True, bounds=[0, 2])
        target = np.array(np.eye(2))
        prob = cp.Problem(cp.Minimize(0), [x == target])

        new_prob = CvxAttr2Constr(prob, reduce_bounds=True)
        assert type(new_prob.apply(prob)[0].constraints[0]) is Inequality
        assert type(new_prob.apply(prob)[0].constraints[1]) is Inequality
        assert type(new_prob.apply(prob)[0].constraints[2]) is Equality

        prob.solve()
        assert np.allclose(x.value, target)
        