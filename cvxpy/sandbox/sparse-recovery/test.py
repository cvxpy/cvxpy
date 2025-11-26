

import matplotlib.pyplot as plt
import numpy as np

import cvxpy as cp

rng = np.random.default_rng(0)

n = 100
m = [50, 56, 62, 68, 74, 80]
k = [30, 34, 38, 42, 46, 50]
T = 3

proba = np.zeros((len(m), len(k)))
proba_l1 = np.zeros((len(m), len(k)))

for time in range(T):
    for kk in k:
        x0 = np.zeros((n, 1))
        ind = rng.permutation(n)
        ind = ind[0:kk]
        x0[ind] = rng.standard_normal((kk, 1)) * 10
        x0 = np.abs(x0)

        for mm in m:
            A = rng.standard_normal((mm, n))
            y = np.dot(A, x0).reshape(-1)

            # sqrt of 0.5-norm minimization
            x_pos = cp.Variable(shape=(n, ), nonneg=True)
            cost = cp.sum(cp.sqrt(x_pos))
            
            prob = cp.Problem(cp.Minimize(cost), [A @ x_pos == y])

            # initialize variable value before solving
            #x_pos.value = np.ones((n))
           
            # initialize to least norm solution
            x0_ls = np.linalg.pinv(A) @ y
            x0_ls = np.maximum(x0_ls, 1)
            x_pos.value = x0_ls

            # try different solvers in case one fails
            result = prob.solve(solver=cp.IPOPT, verbose=True, nlp=True,
                                derivative_test='none', least_square_init_duals='yes')
           

            if result is None:
                x_pos.value = None
#
            norm_diff = cp.pnorm(x_pos - x0, 2).value
            norm_x0 = cp.pnorm(x0, 2).value
            if (
                x_pos.value is not None
                and norm_diff is not None
                and norm_x0 is not None
                and norm_diff / norm_x0 <= 1e-2
            ):
                indm = m.index(mm)
                indk = k.index(kk)
                proba[indm, indk] += 1 / float(T)

            # l1 minimization
            xl1 = cp.Variable((n, 1))
            cost_l1 = cp.pnorm(xl1, 1)
            obj = cp.Minimize(cost_l1)
            constr = [A @ xl1 == y]
            prob_l1 = cp.Problem(obj, constr)
            result_l1 = prob_l1.solve()
#
            norm_diff = cp.pnorm(xl1 - x0, 2).value
            norm_x0 = cp.pnorm(x0, 2).value
            if norm_diff is not None and norm_x0 is not None and norm_diff / norm_x0 <= 1e-2:
                indm = m.index(mm)
                indk = k.index(kk)
                proba_l1[indm, indk] += 1 / float(T)

# validate outputs
assert np.all(proba >= 0) and np.all(proba <= 1)
assert np.all(proba_l1 >= 0) and np.all(proba_l1 <= 1)



fig = plt.figure(figsize=[12, 5], dpi=150)
ax = plt.subplot(1, 2, 1)

plt.xticks(range(len(k)), k)
plt.xlabel("cardinality")
plt.yticks(range(len(m)), m)
plt.ylabel("number of measurements")

a = ax.imshow(proba, interpolation="none")
fig.colorbar(a)
ax.set_title(r"Probability of recovery: $\ell_{1/2}$ norm (DCCP)")
ax = plt.subplot(1, 2, 2)
b = ax.imshow(proba_l1, interpolation="none")
fig.colorbar(b)
plt.xticks(range(len(k)), k)
plt.xlabel("cardinality")
plt.yticks(range(len(m)), m)
plt.ylabel("number of measurements")
ax.set_title(r"Probability of recovery: $\ell_1$ norm (convex)")

# Add overall title
fig.suptitle("Sparse Recovery: Non-convex vs Convex Approaches", fontsize=16, y=1.02)
plt.tight_layout()

plt.savefig("sparse_recovery.pdf")