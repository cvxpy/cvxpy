import numpy as np
import pandas as pd

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
        x = cp.Variable(3, nonneg=True)
        variance = cp.quad_form(x, Q)
        expected_return = r @ x

        prob = cp.Problem(
        cp.Minimize(variance),
        [cp.sum(x) <= 1000, expected_return >= 50]
        )
        return prob


