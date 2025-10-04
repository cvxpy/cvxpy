import numpy as np

import cvxpy as cp


def test_pow_cone_nd_3d(axis):
    """
    A modification of pcp_2. Reformulate

        max  (x**0.2)*(y**0.8) + z**0.4 - x
        s.t. x + y + z/2 == 2
                x, y, z >= 0
    Into

        max  x3 + x4 - x0
        s.t. x0 + x1 + x2 / 2 == 2,

                W := [[x0, x2],
                    [x1, 1.0]]
                z := [x3, x4]
                alpha := [[0.2, 0.4],
                        [0.8, 0.6]]
                (W, z) in PowND(alpha, axis=0)
    """
    x = cp.Variable(shape=(3,), name='x')
    # expect_x = np.array([0.06393515, 0.78320961, 2.30571048])
    hypos = cp.Variable(shape=(2,), name='hypos')
    # expect_hypos = None
    objective = cp.Maximize(cp.sum(hypos) - x[0])
    W = cp.bmat([[x[0], x[2]],
                    [x[1], 1.0]])
    alpha = np.array([[0.2, 0.4],
                        [0.8, 0.6]])
    if axis == 1:
        W = W.T
        alpha = alpha.T

    constraints = [x[0] + x[1] + 0.5 * x[2] == 2, 
                   cp.constraints.PowConeND(W, hypos, alpha, axis=axis)]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=True)


def test_pow_cone_nd(axis):
    """
    A modification of pcp_2. Reformulate

        max  (x**0.2)*(y**0.4)*(w**0.4) + z**0.4 - x
        s.t. x + y + z/2 == 2
                x, y, z >= 0
    Into

        max  x3 + x4 - x0
        s.t. x0 + x1 + x2 / 2 == 2,

                W := [[x0, x3],
                    [x1, 1.0],
                    [x2, 1.0]]
                z := [x4, x5]
                alpha := [[0.2, 0.4],
                        [0.4, 0.3],
                        [0.4, 0.3]]
                (W, z) in PowND(alpha, axis=0)
    """
    x = cp.Variable(shape=(4,), name='x')
    # expect_x = np.array([0.06393515, 0.78320961, 2.30571048])
    hypos = cp.Variable(shape=(2,), name='hypos')
    # expect_hypos = None
    objective = cp.Maximize(cp.sum(hypos) - x[0])
    W = cp.bmat([[x[0], x[3]],
                 [x[1], 1.0],
                 [x[2], 1.0]])
    alpha = np.array([[0.2, 0.4],
                      [0.4, 0.3],
                      [0.4, 0.3]])
    if axis == 1:
        W = W.T
        alpha = alpha.T

    constraints = [x[0] + x[1] + 0.5 * x[2] == 2, 
                   cp.constraints.PowConeND(W, hypos, alpha, axis=axis)]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=True)

test_pow_cone_nd(0)
