import cyipopt
import numpy as np
import pandas as pd
from reduction_classes import HS071, Bounds_Getter

import cvxpy as cp


def portfolio_example():
        # Data
        df = pd.DataFrame({
        'IBM': [93.043, 84.585, 111.453, 99.525, 95.819, 114.708, 111.515,
                113.211, 104.942, 99.827, 91.607, 107.937, 115.590],
        'WMT': [51.826, 52.823, 56.477, 49.805, 50.287, 51.521, 51.531,
                48.664, 55.744, 47.916, 49.438, 51.336, 55.081],
        'SEHI': [1.063, 0.938, 1.000, 0.938, 1.438, 1.700, 2.540, 2.390,
                3.120, 2.980, 1.900, 1.750, 1.800]
        })

        # Compute returns
        returns = df.pct_change().dropna().values
        r = np.mean(returns, axis=0)
        Q = np.cov(returns.T)

        # Single-objective optimization
        x = cp.Variable(3)
        variance = cp.quad_form(x, Q)
        expected_return = r @ x

        prob = cp.Problem(
        cp.Minimize(variance),
        [cp.sum(x) <= 1000, expected_return >= 50]
        )
        return prob

bounds = Bounds_Getter(portfolio_example())
x0 = [10.0, 10.0, 10.0]

nlp = cyipopt.Problem(
   n=len(x0),
   m=len(bounds.cl),
   problem_obj=HS071(bounds.new_problem),
   lb=[0.0, 0.0, 0.0],
   ub=None,
   cl=bounds.cl,
   cu=bounds.cu,
)

nlp.add_option('mu_strategy', 'adaptive')
nlp.add_option('tol', 1e-7)
nlp.add_option('hessian_approximation', "limited-memory")

x, info = nlp.solve(x0)
print(x)
