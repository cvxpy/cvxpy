CVXPY
=====================
[![Build Status](https://travis-ci.org/cvxgrp/cvxpy.png?branch=master)](https://travis-ci.org/cvxgrp/cvxpy)
[![Build status](https://ci.appveyor.com/api/projects/status/jo7tkvc58c3hgfd7?svg=true)](https://ci.appveyor.com/project/StevenDiamond/cvxpy)

**Join the [CVXPY mailing list](https://groups.google.com/forum/#!forum/cvxpy), and use the [issue tracker](https://github.com/cvxgrp/cvxpy/issues) and [StackOverflow](https://stackoverflow.com/questions/tagged/cvxpy) for the best support.**

**The CVXPY documentation is at [cvxpy.org](http://www.cvxpy.org/).**

- [Installation](#installation)
- [Getting started](#getting-started)
- [Issues](#issues)
- [Communication](#communication)
- [Contributing](#contributing)
- [Citing](#citing)
- [Team](#team)

CVXPY is a Python-embedded modeling language for convex optimization problems. It allows you to express your problem in a natural way that follows the math, rather than in the restrictive standard form required by solvers.

For example, the following code solves a least-squares problem where the variable is constrained by lower and upper bounds:

```python
import cvxpy as cp
import numpy

# Problem data.
m = 30
n = 20
numpy.random.seed(1)
A = numpy.random.randn(m, n)
b = numpy.random.randn(m)

# Construct the problem.
x = cp.Variable(n)
objective = cp.Minimize(cp.sum_squares(A*x - b))
constraints = [0 <= x, x <= 1]
prob = cp.Problem(objective, constraints)

# The optimal objective is returned by prob.solve().
result = prob.solve()
# The optimal value for x is stored in x.value.
print(x.value)
# The optimal Lagrange multiplier for a constraint
# is stored in constraint.dual_value.
print(constraints[0].dual_value)
```

CVXPY is not a solver. It relies upon the open source solvers
[ECOS](http://github.com/ifa-ethz/ecos), [SCS](https://github.com/bodono/scs-python),
and [OSQP](https://github.com/oxfordcontrol/osqp). Additional solvers are
[available](https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver),
but must be installed separately.

CVXPY began as a Stanford University research project. It is now developed by
many people, across many institutions and countries.


## Installation
CVXPY is available on pip, and can be installed with
```
pip install cvxpy
```

CVXPY has the following dependencies:

- Python 2.7, 3.4, 3.5, 3.6, or 3.7.
- six
- multiprocess
- OSQP
- ECOS >= 2
- SCS >= 1.1.3
- NumPy >= 1.15
- SciPy >= 1.1.0

For detailed instructions, see the [installation
guide](http://www.cvxpy.org/en/latest/install/index.html).

## Getting started
To get started with CVXPY, check out the following:
* [official CVXPY tutorial](https://www.cvxpy.org/tutorial/index.html)
* [example library](https://www.cvxpy.org/examples/index.html)
* [API reference](https://www.cvxpy.org/api_reference/cvxpy.html)

## Issues
We encourage you to report issues using the [Github tracker](https://github.com/cvxgrp/cvxpy/issues). We welcome all kinds of issues, especially those related to correctness, documentation, performance, and feature requests.

For basic usage questions (e.g., "Why isn't my problem DCP?"), please use [StackOverflow](https://stackoverflow.com/questions/tagged/cvxpy) instead.

## Communication
To communicate with the CVXPY developer community, create a [Github issue](https://github.com/cvxgrp/cvxpy/issues) or use the [CVXPY mailing list](https://groups.google.com/forum/#!forum/cvxpy). Please be respectful in your communications with the CVXPY community, and make sure to abide by our [code of conduct](https://github.com/cvxgrp/cvxpy/blob/master/CODE_OF_CONDUCT.md).

## Contributing
We appreciate all contributions. You don't need to be an expert in convex
optimization to help out.

You should first
install [CVXPY from source](https://www.cvxpy.org/install/index.html#install-from-source).
Here are some simple ways to start contributing immediately:
* Read the CVXPY source code and improve the documentation, or address TODOs
* Enhance the [website documentation](https://github.com/cvxgrp/cvxpy/tree/master/doc)
* Browse the [issue tracker](https://github.com/cvxgrp/cvxpy/issues), and look for issues tagged as "help wanted"
* Polish the [example library](https://github.com/cvxgrp/cvxpy/tree/master/examples)
* Add a [benchmark](https://github.com/cvxgrp/cvxpy/tree/master/cvxpy/tests/test_benchmarks.py)

If you'd like to add a new example to our library, or implement a new feature,
please get in touch with us first to make sure that your priorities align with
ours. 

Contributions should be submitted as [pull requests](https://github.com/cvxgrp/cvxpy/pulls).
A member of the CVXPY development team will review the pull request and guide
you through the contributing process.

Before starting work on your contribution, please read the [contributing guide](https://github.com/cvxgrp/cvxpy/blob/master/CONTRIBUTING.md).

## Citing
If you use CVXPY for academic work, we encourage you to [cite our papers](https://www.cvxpy.org/citing/index.html). If you use CVXPY in industry, we'd love to hear from you as well; feel free to reach out to the developers directly.

## Team
CVXPY is a community project, built from the contributions of many
researchers and engineers.

CVXPY is developed and maintained by [Steven
Diamond](http://web.stanford.edu/~stevend2/) and [Akshay
Agrawal](https://akshayagrawal.com), with many others contributing
significantly. A non-exhaustive list of people who have shaped CVXPY over the
years includes Stephen Boyd, Eric Chu, Robin Verschueren, Bartolomeo Stellato,
Riley Murray, Jaehyun Park, Enzo Busseti, AJ Friend, Judson Wilson, and Chris
Dembia.
