import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cvxpy as cp

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
prob.solve()

print("Single-objective solution:")
print(f"Allocation: {x.value}")
print(f"Variance: {variance.value}")
print(f"Expected return: {expected_return.value}")

# Multi-objective optimization (efficient frontier)
n_points = 50
returns_range = np.linspace(50, r @ np.ones(3) * 1000, n_points)
variances = []
allocations = []

for target_return in returns_range:
    prob = cp.Problem(
        cp.Minimize(variance),
        [cp.sum(x) <= 1000, expected_return >= target_return]
    )
    prob.solve()
    if prob.status == cp.OPTIMAL:
        variances.append(variance.value)
        allocations.append(x.value.copy())

# Plot efficient frontier
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.scatter(variances, returns_range[:len(variances)])
plt.axhline(y=expected_return.value, color='red', linestyle='--', label='Single-objective solution')
plt.axvline(x=variance.value, color='red', linestyle='--')
plt.xlabel('Variance')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.legend()

plt.subplot(2, 1, 2)
allocations_array = np.array(allocations)
plt.bar(range(len(allocations)), allocations_array[:, 0], label='IBM', alpha=0.7)
plt.bar(range(len(allocations)), allocations_array[:, 1], 
        bottom=allocations_array[:, 0], label='WMT', alpha=0.7)
plt.bar(range(len(allocations)), allocations_array[:, 2], 
        bottom=allocations_array[:, 0] + allocations_array[:, 1], label='SEHI', alpha=0.7)
plt.xlabel('Solution #')
plt.ylabel('Investment ($)')
plt.title('Asset Allocation Along Efficient Frontier')
plt.legend()

plt.tight_layout()
plt.show()
