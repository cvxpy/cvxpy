# simple_portfolio_data
import numpy as np
np.random.seed(5)
n = 20
pbar = (np.ones((n, 1)) * .03 +
        np.matrix(np.append(np.random.rand(n - 1, 1), 0)).T * .12)
S = np.matrix(np.random.randn(n, n))
S = S.T * S
S = S / np.max(np.abs(np.diag(S))) * .2
S[:, n - 1] = np.matrix(np.zeros((n, 1)))
S[n - 1, :] = np.matrix(np.zeros((1, n)))
x_unif = np.matrix(np.ones((n, 1))) / n
