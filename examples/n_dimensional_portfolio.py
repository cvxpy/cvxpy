import marimo
import cvxpy as cp
import numpy as np

app = marimo.App()

@app.cell
def __():
    import marimo as mo
    mo.md("# n-Dimensional Portfolio Optimization")
    mo.md("Simple example demonstrating n-dimensional expressions in CVXPY (Markowitz model)")
    return

@app.cell
def __():
    # Number of assets (n-dimensional)
    n = 8
    mu = np.array([0.05, 0.07, 0.04, 0.08, 0.06, 0.09, 0.03, 0.05])
    Sigma = np.random.rand(n, n)
    Sigma = Sigma @ Sigma.T
    Sigma = Sigma / np.max(Sigma) * 0.2

    w = cp.Variable(n)
    ret = mu @ w
    risk = cp.quad_form(w, Sigma)

    prob = cp.Problem(
        cp.Maximize(ret),
        [cp.sum(w) == 1, w >= 0, risk <= 0.04]
    )
    prob.solve()

    print("Optimal weights:", w.value.round(4))
    return