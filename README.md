CVXPY
=====================
[![Build Status](http://github.com/cvxpy/cvxpy/workflows/build/badge.svg?event=push)](https://github.com/cvxpy/cvxpy/actions/workflows/build.yml)
![PyPI - downloads](https://img.shields.io/pypi/dm/cvxpy.svg?label=Pypi%20downloads)
![Conda - downloads](https://img.shields.io/conda/dn/conda-forge/cvxpy.svg?label=Conda%20downloads)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=cvxpy_cvxpy&metric=coverage)](https://sonarcloud.io/summary/new_code?id=cvxpy_cvxpy)
[![Benchmarks](http://img.shields.io/badge/benchmarked%20by-asv-blue.svg?style=flat)](https://cvxpy.github.io/benchmarks/)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/cvxpy/cvxpy/badge)](https://api.securityscorecards.dev/projects/github.com/cvxpy/cvxpy)

**The CVXPY documentation is at [cvxpy.org](http://www.cvxpy.org/).**

*We are building a CVXPY community on [Discord](https://discord.gg/4urRQeGBCr). Join the conversation! For issues and long-form discussions, use [Github Issues](https://github.com/cvxpy/cvxpy/issues) and [Github Discussions](https://github.com/cvxpy/cvxpy/discussions).*

**Contents**
- [Installation](#installation)
- [Getting started](#getting-started)
- [Issues](#issues)
- [Community](#community)
- [Contributing](#contributing)
- [Team](#team)
- [Citing](#citing)


CVXPY is a Python-embedded modeling language for convex optimization problems. It allows you to express your problem in a natural way that follows the math, rather than in the restrictive standard form required by solvers.

For example, the following code solves a least-squares problem where the variable is constrained by lower and upper bounds:

```python3
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
objective = cp.Minimize(cp.sum_squares(A @ x - b))
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

With CVXPY, you can model
* convex optimization problems,
* mixed-integer convex optimization problems,
* geometric programs, and
* quasiconvex programs.

CVXPY is not a solver. It relies upon the open source solvers
[ECOS](http://github.com/ifa-ethz/ecos), [SCS](https://github.com/bodono/scs-python),
and [OSQP](https://github.com/oxfordcontrol/osqp). Additional solvers are
[available](https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver),
but must be installed separately.

CVXPY began as a Stanford University research project. It is now developed by
many people, across many institutions and countries.


## Installation
CVXPY is available on PyPI, and can be installed with
```
pip install cvxpy
```

CVXPY can also be installed with conda, using
```
conda install -c conda-forge cvxpy
```

CVXPY has the following dependencies:

- Python >= 3.8
- Clarabel >= 0.5.0
- OSQP >= 0.6.2
- ECOS >= 2
- SCS >= 3.0
- NumPy >= 1.15
- SciPy >= 1.1.0

For detailed instructions, see the [installation
guide](https://www.cvxpy.org/install/index.html).

## Getting started
To get started with CVXPY, check out the following:
* [official CVXPY tutorial](https://www.cvxpy.org/tutorial/index.html)
* [example library](https://www.cvxpy.org/examples/index.html)
* [API reference](https://www.cvxpy.org/api_reference/cvxpy.html)

## Issues
We encourage you to report issues using the [Github tracker](https://github.com/cvxpy/cvxpy/issues). We welcome all kinds of issues, especially those related to correctness, documentation, performance, and feature requests.

For basic usage questions (e.g., "Why isn't my problem DCP?"), please use [StackOverflow](https://stackoverflow.com/questions/tagged/cvxpy) instead.

## Community
The CVXPY community consists of researchers, data scientists, software engineers, and students from all over the world. We welcome you to join us!

* To chat with the CVXPY community in real-time, join us on [Discord](https://discord.gg/4urRQeGBCr).
* To have longer, in-depth discussions with the CVXPY community, use [Github Discussions](https://github.com/cvxpy/cvxpy/discussions).
* To share feature requests and bug reports, use [Github Issues](https://github.com/cvxpy/cvxpy/issues).

Please be respectful in your communications with the CVXPY community, and make sure to abide by our [code of conduct](https://github.com/cvxpy/cvxpy/blob/master/CODE_OF_CONDUCT.md).

## Contributing
We appreciate all contributions. You don't need to be an expert in convex
optimization to help out.

You should first
install [CVXPY from source](https://www.cvxpy.org/install/index.html#install-from-source).
Here are some simple ways to start contributing immediately:
* Read the CVXPY source code and improve the documentation, or address TODOs
* Enhance the [website documentation](https://github.com/cvxpy/cvxpy/tree/master/doc)
* Browse the [issue tracker](https://github.com/cvxpy/cvxpy/issues), and look for issues tagged as "help wanted"
* Polish the [example library](https://github.com/cvxpy/cvxpy/tree/master/examples)
* Add a [benchmark](https://github.com/cvxpy/cvxpy/tree/master/cvxpy/tests/test_benchmarks.py)

If you'd like to add a new example to our library, or implement a new feature,
please get in touch with us first to make sure that your priorities align with
ours. 

Contributions should be submitted as [pull requests](https://github.com/cvxpy/cvxpy/pulls).
A member of the CVXPY development team will review the pull request and guide
you through the contributing process.

Before starting work on your contribution, please read the [contributing guide](https://github.com/cvxpy/cvxpy/blob/master/CONTRIBUTING.md).

## Team
CVXPY is a community project, built from the contributions of many
researchers and engineers.

CVXPY is developed and maintained by [Steven
Diamond](https://stevendiamond.me/), [Akshay
Agrawal](https://akshayagrawal.com), [Riley Murray](https://rileyjmurray.wordpress.com/), 
[Philipp Schiele](https://www.philippschiele.com/),
and [Bartolomeo Stellato](https://stellato.io/), with many others contributing
significantly. A non-exhaustive list of people who have shaped CVXPY over the
years includes Stephen Boyd, Eric Chu, Robin Verschueren, Michael Sommerauer,
Jaehyun Park, Enzo Busseti, AJ Friend, Judson Wilson, and Chris Dembia.

For more information about the team and our processes, see our [governance document](https://github.com/cvxpy/org/blob/main/governance.md).

## Citing
If you use CVXPY for academic work, we encourage you to [cite our papers](https://www.cvxpy.org/citing/index.html). If you use CVXPY in industry, we'd love to hear from you as well, on Discord or over email.
