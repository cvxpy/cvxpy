import numpy as np
import scipy.sparse.linalg as sparla
from numpy.random import PCG64, Generator

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

    # We simulate a scenario where a matrix is PSD but a ArpackNoConvergence is raised.
    # With the numpy PCG64-based random number generator, this happens with seed 119.
    # We test a range of seeds to make sure that this scenario is not always triggered.

    failures = set()
    for seed in range(115, 120):
        rng = Generator(PCG64(seed))

        P = rng.standard_normal((n, n))
        P = P.T @ P

        try:
            cp.Constant(P).is_psd()
        except sparla.ArpackNoConvergence as e:
            assert "CVXPY note" in str(e)
            failures.add(seed)
    assert failures == {119}

    assert psd_wrap(cp.Constant(P)).is_psd()
