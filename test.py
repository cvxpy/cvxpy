import cvxpy as cp
import numpy as np

def impute_distmat(D, W, alpha=1, verbose=False):
    # Define V and e
    n = D.shape[0]
    x = -1/(n + np.sqrt(n))
    y = -1/np.sqrt(n)
    V = np.ones((n, n-1))
    V[0, :] *= y
    V[1:, :] *= x
    V[1:, :] += np.eye(n-1)
    e = np.ones((n, 1))

    # Solve optimization problem
    G = cp.Variable((n-1, n-1), PSD=True)
    objective = cp.Maximize(cp.trace(G) - alpha*cp.norm(cp.multiply(W, cp.kron(e, cp.reshape(cp.diag(V@G@V.T), (1, n))) + cp.kron(e.T, cp.reshape(cp.diag(V@G@V.T), (n, 1))) - 2*V@G@V.T - D), p='fro'))
    prob = cp.Problem(objective, [])
    prob.solve(verbose=verbose)

    if G.value is None:
        D_rc = None
    else:
        B = V@G.value@V.T
        D_rc = np.kron(e, np.reshape(np.diag(B), (1, n))) + np.kron(e.T, np.reshape(np.diag(B), (n, 1))) - 2*B

    return prob.status, prob.value, D_rc

#############################################
# Construct D
n = 50
np.random.seed(0)
points = np.random.rand(5, n)
xtx = points.T@points
xtxd = np.diag(xtx)
e = np.ones((n, ))
D = np.outer(e, xtxd) - 2*xtx + np.outer(xtxd, e)
# Construct W
W = np.ones((n, n))
# Optimization Problem
opt_status, opt_value, D_rc = impute_distmat(D, W, verbose=True)
