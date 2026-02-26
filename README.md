# DNLP — Disciplined Nonlinear Programming
The DNLP package is an extension of [CVXPY](https://www.cvxpy.org/) to general nonlinear programming (NLP).
DNLP allows smooth functions to be freely mixed with nonsmooth convex and concave functions,
with some rules governing how the nonsmooth functions can be used. For details, see our paper [Disciplined Nonlinear Programming](https://web.stanford.edu/~boyd/papers/dnlp.html).

---
## Installation
The installation consists of two steps.

#### Step 1: Install IPOPT
DNLP requires an NLP solver. The recommended solver is [Ipopt](https://coin-or.github.io/Ipopt/). First install the IPOPT system library, then install the Python interface [cyipopt](https://github.com/mechmotum/cyipopt):
```bash
# Ubuntu/Debian
sudo apt-get install coinor-libipopt-dev

# macOS
brew install ipopt
```
Then install the Python interface:
```bash
pip install cyipopt
```

#### Step 2: Install DNLP
DNLP is installed by cloning this repository and installing it locally:
```bash
git clone https://github.com/cvxgrp/DNLP.git
cd DNLP
pip install .
```

---
## Example
Below we give a toy example where we maximize a convex quadratic function subject to a nonlinear equality constraint.  Many more examples, including the ones in the paper, can be found at [DNLP-examples](https://github.com/cvxgrp/dnlp-examples).
```python
import cvxpy as cp
import numpy as np
import cvxpy as cp

# problem data
np.random.seed(0)
n = 3
A = np.random.randn(n, n)
A = A.T @ A

# formulate optimization problem
x = cp.Variable(n)
obj = cp.Maximize(cp.quad_form(x, A))
constraints = [cp.sum_squares(x) == 1]

# initialize and solve
x.value = np.ones(n)
prob = cp.Problem(obj, constraints)
prob.solve(nlp=True, verbose=True)
print("Optimal value from DNLP: ", prob.value)

# the optimal value for this toy problem can also be found by computing the maximum eigenvalue of A
eigenvalues  = np.linalg.eigvalsh(A)
print("Maximum eigenvalue:      " , np.max(eigenvalues))
```

---
## Supported Solvers
| Solver | License | Installation |
|--------|---------|--------------|
| [IPOPT](https://github.com/coin-or/Ipopt) | EPL-2.0 | Install system IPOPT (see above), then `pip install cyipopt` |
| [Knitro](https://www.artelys.com/solvers/knitro/) | Commercial | `pip install knitro` (requires license) |

---
## Differentiation Engine
DNLP uses [SparseDiffPy](https://github.com/SparseDifferentiation/SparseDiffPy) as its differentiation engine. SparseDiffPy is a Python wrapper around the [SparseDiffEngine](https://github.com/SparseDifferentiation/SparseDiffEngine) C library, and is installed automatically as a dependency of DNLP.

SparseDiffPy builds an expression tree from the CVXPY problem and computes exact sparse gradients, Jacobians, and Hessians required by the NLP solvers.
