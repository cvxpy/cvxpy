# This is just a placeholder file to show how the prototype API works.
# We'll delete this file before the PR is merged.
import cvxpy.cvxcore.python.sparsecholesky as spchol
import scipy.sparse as spar
import numpy as np
import scipy.linalg as la
import cvxpy.utilities.linalg

np.random.seed(0)

n = 5

inrows = []
incols = []
invals = []
for i in range(n):
    inrows.append(i)
    incols.append(i)
    invals.append(2.0)
    if i < n - 1:
        inrows.append(i)
        incols.append(i+1)
        invals.append(0.1)
    if i > 0:
        inrows.append(i)
        incols.append(i-1)
        invals.append(0.1)

G = spar.csr_matrix((invals, (inrows, incols)), shape=(n, n))

Lp = cvxpy.utilities.linalg.sparse_cholesky(G)
print(la.norm((Lp @ Lp.T - G).toarray()))

G = G - 1.95 * spar.eye(n)
