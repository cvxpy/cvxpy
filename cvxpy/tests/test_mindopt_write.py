import os
import unittest

import numpy as np

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS


@unittest.skipUnless('MINDOPT' in INSTALLED_SOLVERS, 'MINDOPT is not installed.')
def test_write(tmpdir):
    """
    Test the MindOpt model.write().
    """
    # 测试成功
    filename = "mindopt_model.lp"
    # filename = "mindopt_model.mps"
    path = os.path.join(tmpdir, filename)

    m = 20
    n = 15
    np.random.seed(0)
    A = np.random.randn(m, n)
    b = np.random.randn(m)

    x = cp.Variable(n)
    cost = cp.sum_squares(A @ x - b)
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve(solver=cp.MINDOPT,
                verbose=True,
                save_file=path)

    # Print result.
    print("\nThe optimal value is", prob.value)
    print("The optimal x is")
    print(x.value)
    print("The norm of the residual is ", cp.norm(A @ x - b, p=2).value)

    assert os.path.exists(path)

# test_write("./resources")

@unittest.skipUnless('MINDOPT' in INSTALLED_SOLVERS, 'MINDOPT is not installed.')
def test_basic_lp():
    '''
    lp example test
    '''

    # Generate a random non-trivial linear program.
    m = 15
    n = 10
    np.random.seed(1)
    s0 = np.random.randn(m)
    lamb0 = np.maximum(-s0, 0)
    s0 = np.maximum(s0, 0)
    x0 = np.random.randn(n)
    A = np.random.randn(m, n)
    b = A @ x0 + s0
    c = -A.T @ lamb0

    # Define and solve the CVXPY problem.
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(c.T @ x),
                      [A @ x <= b])
    prob.solve(solver=cp.MINDOPT,
                verbose=True)

    # Print result.
    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    print("A dual solution is")
    print(prob.constraints[0].dual_value)

# test_basic_lp()
@unittest.skipUnless('MINDOPT' in INSTALLED_SOLVERS, 'MINDOPT is not installed.')
def test_basic_qp():
    '''
    测试qp基础
    '''
    # Import packages.
    import cvxpy as cp
    import numpy as np

    # Generate a random non-trivial quadratic program.
    m = 15
    n = 10
    p = 5
    np.random.seed(1)
    P = np.random.randn(n, n)
    P = P.T @ P
    q = np.random.randn(n)
    G = np.random.randn(m, n)
    h = G @ np.random.randn(n)
    A = np.random.randn(p, n)
    b = np.random.randn(p)

    # Define and solve the CVXPY problem.
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, P) + q.T @ x),
                      [G @ x <= h,
                       A @ x == b])
    prob.solve(solver=cp.MINDOPT,
                verbose=True)

    # Print result.
    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    print("A dual solution corresponding to the inequality constraints is")
    print(prob.constraints[0].dual_value)

# test_basic_qp()

@unittest.skipUnless('MINDOPT' in INSTALLED_SOLVERS, 'MINDOPT is not installed.')
def test_basic_socp():
    '''
    测试Second-order cone program¶
    '''
    # Import packages.
    import cvxpy as cp
    import numpy as np

    # Generate a random feasible SOCP.
    m = 3
    n = 10
    p = 5
    n_i = 5
    np.random.seed(2)
    f = np.random.randn(n)
    A = []
    b = []
    c = []
    d = []
    x0 = np.random.randn(n)
    for i in range(m):
        A.append(np.random.randn(n_i, n))
        b.append(np.random.randn(n_i))
        c.append(np.random.randn(n))
        d.append(np.linalg.norm(A[i] @ x0 + b, 2) - c[i].T @ x0)
    F = np.random.randn(p, n)
    g = F @ x0

    # Define and solve the CVXPY problem.
    x = cp.Variable(n)
    # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
    soc_constraints = [
        cp.SOC(c[i].T @ x + d[i], A[i] @ x + b[i]) for i in range(m)
    ]
    prob = cp.Problem(cp.Minimize(f.T @ x),
                      soc_constraints + [F @ x == g])
    prob.solve(solver=cp.MINDOPT,
                verbose=True)

    # Print result.
    print("The optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    for i in range(m):
        print("SOC constraint %i dual variable solution" % i)
        print(soc_constraints[i].dual_value)

# test_basic_socp()

@unittest.skipUnless('MINDOPT' in INSTALLED_SOLVERS, 'MINDOPT is not installed.')
def test_basic_milp():
    '''
    测试MILP
    '''
    # Generate a random problem
    np.random.seed(0)
    m, n = 40, 25

    A = np.random.rand(m, n)
    b = np.random.randn(m)
    # Construct a CVXPY problem
    x = cp.Variable(n, integer=True)
    objective = cp.Minimize(cp.sum_squares(A @ x - b))
    prob = cp.Problem(objective)
    prob.solve(solver=cp.MINDOPT,
                verbose=True)

    print("Status: ", prob.status)
    print("The optimal value is", prob.value)
    print("A solution x is")
    print(x.value)

# test_basic_milp()

@unittest.skipUnless('MINDOPT' in INSTALLED_SOLVERS, 'MINDOPT is not installed.')
def test_basic_miqp():
    '''
    miqp example test
    '''
    print("Generate a MIQP Problem")

    # Generate a random problem
    np.random.seed(0)
    m, n = 40, 25

    A = np.random.rand(m, n)
    b = np.random.randn(m)

    # Construct a CVXPY problem
    x = cp.Variable(n, integer=True)
    objective = cp.Minimize(cp.sum_squares(A @ x - b))
    prob = cp.Problem(objective)
    prob.solve(solver=cp.MINDOPT,
                verbose=True)

    print("Status: ", prob.status)
    print("The optimal value is", prob.value)
    print("A solution x is")
    print(x.value)


test_basic_miqp()

@unittest.skipUnless('MINDOPT' in INSTALLED_SOLVERS, 'MINDOPT is not installed.')
def test_basic_miqcp():
    '''
    miqcp example test
    '''
    
    np.random.seed(42)
    
    # Expected return and covariance matrix
    expected_returns = np.array([0.1, 0.12, 0.08, 0.15, 0.11])
    cov_matrix = np.array([
        [0.04, 0.01, 0.005, 0.02, 0.01],
        [0.01, 0.09, 0.003, 0.015, 0.02],
        [0.005, 0.003, 0.06, 0.01, 0.005],
        [0.02, 0.015, 0.01, 0.16, 0.03],
        [0.01, 0.02, 0.005, 0.03, 0.11]
    ])
    
    # Define variables
    # Continuous variables: investment proportions
    weights_continuous = cp.Variable(3)  # First 3 assets can be partially invested
    # Integer variables: number of lots (assuming 100 shares per lot)
    weights_integer = cp.Variable(2, integer=True)  # Last 2 assets must be invested in whole lots
    
    # Combined variables
    weights = cp.hstack([weights_continuous, weights_integer * 100])  # Assume integers represent lots
    
    # Objective function: minimize portfolio variance (risk)
    portfolio_variance = cp.quad_form(weights, cov_matrix)
    objective = cp.Minimize(portfolio_variance)
    
    # Constraints
    constraints = [
        cp.sum(weights) == 1000,  # Total investment amount is 1000
        weights >= 0,             # All investments are non-negative
        
        # Return constraint: expected return at least 12%
        expected_returns @ weights >= 120,  # 1000 * 0.12 = 120
        
        # Integer variable constraints
        weights_integer >= 0,     # Integer variables are non-negative
        weights_continuous >= 0,  # Continuous variables are non-negative
    ]
    
    # Create and solve the problem
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.MINDOPT)
    except:
        try:
            problem.solve(solver=cp.GUROBI)
        except:
            problem.solve(solver=cp.ECOS_BB)
    
    # Output results
    print("=== Portfolio Optimization Results ===")
    print("Solution status:", problem.status)
    assert problem.status == 'optimal'

    print("Minimum risk (variance):", problem.value)
    assert abs(problem.value - 36173.59375) < 1e-4

    print("Investment allocation:")
    print(f"  Asset 1 (continuous): {weights_continuous[0].value:.6f}")
    assert abs(weights_continuous[0].value - 279.687500) < 1e-1

    print(f"  Asset 2 (continuous): {weights_continuous[1].value:.6f}")
    assert abs(weights_continuous[1].value - 260.156250) < 1e-1

    print(f"  Asset 3 (continuous): {weights_continuous[2].value:.6f}")
    assert abs(weights_continuous[2].value - 60.156250) < 1e-1

    print(f"  Asset 4 (integer lots): {weights_integer[0].value} lots")
    assert abs(weights_integer[0].value - 3) < 1e-4

    print(f"  Asset 5 (integer lots): {weights_integer[1].value} lots")
    assert abs(weights_integer[1].value - 1) < 1e-4
    
    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        total_investment = sum(weights.value)
        expected_return = expected_returns @ weights.value
        print(f"\nVerification:")
        print(f"  Total investment: {total_investment:.6f} (should = 1000)")
        assert abs(total_investment - 1000) < 1e-4

        print(f"  Expected return: {expected_return:.6f} (should ≥ 120)")
        assert abs(expected_return - 120) < 1e-4

        print(f"  Actual return rate: {expected_return/1000*100:.6f}%")
        assert abs(expected_return/1000*100 - 12) < 1e-4


test_basic_miqcp()

@unittest.skipUnless('MINDOPT' in INSTALLED_SOLVERS, 'MINDOPT is not installed.')
def test_basic_control():
    '''
    测试基础案例control
    '''
    np.random.seed(1)
    n = 8
    m = 2
    T = 50
    alpha = 0.2
    beta = 3
    A = np.eye(n) - alpha * np.random.rand(n, n)
    B = np.random.randn(n, m)
    x_0 = beta * np.random.randn(n)

    x = cp.Variable((n, T + 1))
    u = cp.Variable((m, T))

    cost = 0
    constr = []
    for t in range(T):
        cost += cp.sum_squares(x[:, t + 1]) + cp.sum_squares(u[:, t])
        constr += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t], cp.norm(u[:, t], "inf") <= 1]
    # sums problem objectives and concatenates constraints.
    constr += [x[:, T] == 0, x[:, 0] == x_0]
    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve(solver=cp.MINDOPT,
                verbose=True)

# test_basic_control()
@unittest.skipUnless('MINDOPT' in INSTALLED_SOLVERS, 'MINDOPT is not installed.')
def test_finance_portfolio1():
    '''
    案例portfolio测试,
    result:https://colab.research.google.com/github/cvxgrp/cvx_short_course/blob/master/book/docs/applications/notebooks/portfolio_optimization.ipynb#scrollTo=OykuoyCY10aA
    '''
    np.random.seed(1)
    n = 10
    mu = np.abs(np.random.randn(n, 1))
    Sigma = np.random.randn(n, n)
    Sigma = Sigma.T.dot(Sigma)
    w = cp.Variable(n)
    gamma = cp.Parameter(nonneg=True)
    ret = mu.T @ w
    risk = cp.quad_form(w, Sigma)
    prob = cp.Problem(cp.Maximize(ret - gamma * risk), [cp.sum(w) == 1, w >= 0])
    # Compute trade-off curve.
    SAMPLES = 100
    risk_data = np.zeros(SAMPLES)
    ret_data = np.zeros(SAMPLES)
    gamma_vals = np.logspace(-2, 3, num=SAMPLES)
    for i in range(SAMPLES):
        gamma.value = gamma_vals[i]
        # test
        prob.solve(solver=cp.MINDOPT, verbose=True)
        risk_data[i] = cp.sqrt(risk).value
        ret_data[i] = ret.value
    import matplotlib.pyplot as plt

    markers_on = [29, 40]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(risk_data, ret_data, "g-")
    for marker in markers_on:
        plt.plot(risk_data[marker], ret_data[marker], "bs")
        ax.annotate(
            r"$\gamma = %.2f$" % gamma_vals[marker],
            xy=(risk_data[marker] + 0.08, ret_data[marker] - 0.03),
        )
    for i in range(n):
        plt.plot(cp.sqrt(Sigma[i, i]).value, mu[i], "ro")
    plt.xlabel("Standard deviation")
    plt.ylabel("Return")
    plt.show()

# test_finance_portfolio1()

@unittest.skipUnless('MINDOPT' in INSTALLED_SOLVERS, 'MINDOPT is not installed.')
def test_finance_portfolio2():
    '''
    测试portfolio optimization问题2
    '''
    import scipy.sparse as sp
    # Generate data for factor model.
    n = 600
    m = 30
    np.random.seed(1)
    mu = np.abs(np.random.randn(n, 1))
    Sigma_tilde = np.random.randn(m, m)
    Sigma_tilde = Sigma_tilde.T.dot(Sigma_tilde)
    D = sp.diags(np.random.uniform(0, 0.9, size=n))
    F = np.random.randn(n, m)

    # Factor model portfolio optimization.
    w = cp.Variable(n)
    f = cp.Variable(m)
    gamma = cp.Parameter(nonneg=True)
    Lmax = cp.Parameter()
    ret = mu.T @ w
    risk = cp.quad_form(f, Sigma_tilde) + cp.sum_squares(np.sqrt(D) @ w)
    prob_factor = cp.Problem(
        cp.Maximize(ret - gamma * risk),
        [cp.sum(w) == 1, f == F.T @ w, cp.norm(w, 1) <= Lmax],
    )

    # Solve the factor model problem.
    Lmax.value = 2
    gamma.value = 0.1
    # Solver details: Solver reached iteration limit.
    prob_factor.solve(solver=cp.MINDOPT, verbose=True)

# 求解达到上限
# test_finance_portfolio2()

@unittest.skipUnless('MINDOPT' in INSTALLED_SOLVERS, 'MINDOPT is not installed.')
def test_finance_portfolio3():
    '''
    测试portfolio optimization
    '''

    np.random.seed(1)
    n = 10
    mu = np.abs(np.random.randn(n, 1))
    Sigma = np.random.randn(n, n)
    Sigma = Sigma.T.dot(Sigma)
    w = cp.Variable(n)
    gamma = cp.Parameter(nonneg=True)
    ret = mu.T @ w
    risk = cp.quad_form(w, Sigma)
    Lmax = cp.Parameter()
    prob = cp.Problem(
        cp.Maximize(ret - gamma * risk), [cp.sum(w) == 1, cp.norm(w, 1) <= Lmax]
    )
    # Compute trade-off curve for each leverage limit.
    L_vals = [1, 2, 4]
    SAMPLES = 100
    risk_data = np.zeros((len(L_vals), SAMPLES))
    ret_data = np.zeros((len(L_vals), SAMPLES))
    gamma_vals = np.logspace(-2, 3, num=SAMPLES)
    w_vals = []
    for k, L_val in enumerate(L_vals):
        for i in range(SAMPLES):
            Lmax.value = L_val
            gamma.value = gamma_vals[i]
            prob.solve(solver=cp.MINDOPT, verbose=True)
            risk_data[k, i] = cp.sqrt(risk).value
            ret_data[k, i] = ret.value
    import matplotlib.pyplot as plt
    # Plot trade-off curves for each leverage limit.
    for idx, L_val in enumerate(L_vals):
        plt.plot(risk_data[idx, :], ret_data[idx, :], label=r"$L^{\max}$ = %d" % L_val)
    for w_val in w_vals:
        w.value = w_val
        plt.plot(cp.sqrt(risk).value, ret.value, "bs")
    plt.xlabel("Standard deviation")
    plt.ylabel("Return")
    plt.legend(loc="lower right")
    plt.show()

# 达到上限迭代次数4000
# test_finance_portfolio3()

@unittest.skipUnless('MINDOPT' in INSTALLED_SOLVERS, 'MINDOPT is not installed.')
def test_finance_portfolio4():
    '''
    测试portfolio optimization
    '''
    np.random.seed(1)
    n = 10
    mu = np.abs(np.random.randn(n, 1))
    Sigma = np.random.randn(n, n)
    Sigma = Sigma.T.dot(Sigma)
    w = cp.Variable(n)
    gamma = cp.Parameter(nonneg=True)
    ret = mu.T @ w
    risk = cp.quad_form(w, Sigma)
    Lmax = cp.Parameter()
    # Portfolio optimization with a leverage limit and a bound on risk.
    prob = cp.Problem(cp.Maximize(ret), [cp.sum(w) == 1, cp.norm(w, 1) <= Lmax, risk <= 2])
    # Compute solution for different leverage limits.
    L_vals = [1, 2, 4]
    w_vals = []
    for k, L_val in enumerate(L_vals):
        Lmax.value = L_val
        prob.solve()
        w_vals.append(w.value)

    import matplotlib.pyplot as plt
    # Plot bar graph of holdings for different leverage limits.
    colors = ["b", "g", "r"]
    indices = np.argsort(mu.flatten())
    for idx, L_val in enumerate(L_vals):
        plt.bar(
            np.arange(1, n + 1) + 0.25 * idx - 0.375,
            w_vals[idx][indices],
            color=colors[idx],
            label=r"$L^{\max}$ = %d" % L_val,
            width=0.25,
        )
    plt.ylabel(r"$w_i$", fontsize=16)
    plt.xlabel(r"$i$", fontsize=16)
    plt.xlim([1 - 0.375, 10 + 0.375])
    plt.xticks(np.arange(1, n + 1))
    plt.show()

# test_finance_portfolio4()

@unittest.skipUnless('MINDOPT' in INSTALLED_SOLVERS, 'MINDOPT is not installed.')
def test_finance_GINI_Portfolio():
    '''
    测试案例GINI Portfolio
    '''
    import pandas as pd
    import scipy.stats as st
    from timeit import default_timer as timer
    from datetime import timedelta
    import warnings

    warnings.filterwarnings("ignore")

    def gini(mu, returns, D, assets, lift=0):
        (T, N) = returns.shape

        d = cp.Variable((int(T * (T - 1) / 2), 1))
        w = cp.Variable((N, 1))
        constraints = []

        if lift in ['Murray', 'Yitzhaki']:  # use Murray's reformulation
            if lift == 'Murray':
                ret_w = cp.Variable((T, 1))
                constraints.append(ret_w == returns @ w)
                mat = np.zeros((d.shape[0], T))
                """ 
                We need to create a vector that has the following entries:
                    ret_w[i] - ret_w[j]
                for j in range(T), for i in range(j+1, T).
                We do this by building a numpy array of mostly 0's and 1's.
                (It would be better to use SciPy sparse matrix objects.)
                """
                ell = 0
                for j in range(T):
                    for i in range(j + 1, T):
                        # write to mat so that (mat @ ret_w)[ell] == var_i - var_j
                        mat[ell, i] = 1
                        mat[ell, j] = -1
                        ell += 1
                all_pairs_ret_diff = mat @ ret_w
            elif lift == 'Yitzhaki':  # use the original formulation
                all_pairs_ret_diff = D @ w

            constraints += [d >= all_pairs_ret_diff,
                            d >= -all_pairs_ret_diff,
                            w >= 0,
                            cp.sum(w) == 1,
                            ]

            risk = cp.sum(d) / ((T - 1) * T)

        elif lift == 'Cajas':
            a = cp.Variable((T, 1))
            b = cp.Variable((T, 1))
            y = cp.Variable((T, 1))

            owa_w = []
            for i in range(1, T + 1):
                owa_w.append(2 * i - 1 - T)
            owa_w = np.array(owa_w) / (T * (T - 1))

            constraints = [returns @ w == y,
                           w >= 0,
                           cp.sum(w) == 1]

            for i in range(T):
                constraints += [a[i] + b >= cp.multiply(owa_w[i], y)]

            risk = cp.sum(a + b)

        objective = cp.Minimize(risk * 1000)
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.MINDOPT,
                       verbose=True)
            w = pd.DataFrame(w.value)
            w.index = assets
            w = w / w.sum()
        except:
            w = None
        return w

    rs = np.random.RandomState(123)

    sizes = [100]
    data = {}
    weights = {}
    lifts = ['Yitzhaki', 'Murray', 'Cajas']
    k = 0
    # T = [200, 500, 700, 1000]
    T = [200]

    for t in T:
        for n in sizes:

            cov = rs.rand(n, n) * 1.5 - 0.5
            cov = cov @ cov.T / 1000 + np.diag(rs.rand(n) * 0.7 + 0.3) / 1000
            mean = np.zeros(n) + 1 / 1000

            Y = st.multivariate_normal.rvs(mean=mean, cov=cov, size=t, random_state=rs)
            Y = pd.DataFrame(Y)
            assets = ['Asset ' + str(i) for i in range(1, n + 1)]
            mu = Y.mean().to_numpy()
            returns = Y.to_numpy()

            D = np.array([]).reshape(0, len(assets))
            for j in range(0, returns.shape[0] - 1):
                D = np.concatenate((D, returns[j + 1:] - returns[j, :]), axis=0)
            print(1)
            for lift in lifts:
                name = str(lift) + '-' + str(t) + '-' + str(n)
                print(lift, name)
                data[name] = []
                weights[name] = []
                if t >= 700 and lift == 'Yitzhaki':
                    continue
                else:
                    start = timer()
                    w = gini(mu, returns, D, assets, lift=lift)
                    end = timer()
                    data[name].append(timedelta(seconds=end - start).total_seconds())
                    weights[name].append(w)

                k += 1
                print(name)

# test_finance_GINI_Portfolio()