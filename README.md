# DNLP — Disciplined Nonlinear Programming
The DNLP package is an extension of [CVXPY](https://www.cvxpy.org/) to general nonlinear programming (NLP).
DNLP allows smooth functions to be freely mixed with nonsmooth convex and concave functions, 
with some rules governing how the nonsmooth functions can be used. For details, see our paper [Disciplined Nonlinear Programming](XXX).

---
## Installation
The installation consists of two steps.

#### Step 1: Install IPOPT via Conda
DNLP requires an NLP solver. The recommended solver is [Ipopt](XXX), which can be installed together with its Python interface [cyipopt](https://github.com/mechmotum/cyipopt):
```bash
conda install -c conda-forge cyipopt
```
Installing cyipopt via pip may lead to issues, so we strongly recommend using the Conda installation above, even if the rest of your environment uses pip.

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
| [IPOPT](https://github.com/coin-or/Ipopt) | EPL-2.0 | `conda install -c conda-forge cyipopt` |
| [Knitro](https://www.artelys.com/solvers/knitro/) | Commercial | `pip install knitro` (requires license) |
