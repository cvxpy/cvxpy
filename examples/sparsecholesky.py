# This is just a placeholder file to show how the prototype API works.
# We'll delete this file before the PR is merged.
import scipy.sparse as spar
import numpy as np
import cvxpy.cvxcore.python.sparsecholesky as spchol

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

inrows = spchol.IntVector(inrows)
incols = spchol.IntVector(incols)
invals = spchol.DoubleVector(invals)
outpivs = spchol.IntVector(0)
outrows = spchol.IntVector(0)
outcols = spchol.IntVector(0)
outvals = spchol.DoubleVector(0)

spchol.sparse_chol_from_vecs(
    n, inrows, incols, invals,
    outpivs, outrows, outcols, outvals
)

outpivs = list(outpivs)
L = spar.csc_matrix((outvals, (outrows, outcols)), shape=(n, n))
print(outpivs)
print(L.toarray())
