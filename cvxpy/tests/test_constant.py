import numpy as np
import pytest
import scipy.sparse.linalg as sparla

import cvxpy as cp
from cvxpy import psd_wrap


def test_is_psd() -> None:

    n = 50

    # trivial cases
    psd = np.eye(n)
    nsd = -np.eye(n)

    assert cp.Constant(psd).is_psd()
    assert not cp.Constant(psd).is_nsd()

    assert cp.Constant(nsd).is_nsd()
    assert not cp.Constant(nsd).is_psd()

    np.random.seed(97)

    P = np.random.randn(n, n)
    P = P.T @ P

    with pytest.raises(sparla.ArpackNoConvergence, match="CVXPY note"):
        cp.Constant(P).is_psd()

    assert psd_wrap(cp.Constant(P)).is_psd()
