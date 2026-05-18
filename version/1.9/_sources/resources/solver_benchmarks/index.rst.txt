Solver Benchmarks
=================

The `solver-benchmarks <https://github.com/cvxpy/solver-benchmarks>`_ project
collects benchmark problems and results to inform CVXPY's default solver
selection. By running a shared suite of problems across many machines and solver
versions, we can make data-driven decisions about which solvers to recommend as
defaults for each problem type (LP, QP, MIP, SOCP, SDP).

Why this project exists
-----------------------

CVXPY supports many solvers, but choosing good defaults requires real
performance data across a wide range of problems and environments.
The solver-benchmarks project provides that data so CVXPY can recommend the
best solver for each problem type out of the box.

Phase 1: Building benchmark problems (current)
-----------------------------------------------

We are collecting benchmark problems from the community. Ideal problems:

- Take about **one second** to solve.
- Are based on **real-world** applications.
- Use **real data** when possible.
- Cover the full range of problem types: LP, QP, MIP, SOCP, SDP.

If you have a problem that fits these criteria, please contribute it!

Phase 2: Running benchmarks and contributing results
----------------------------------------------------

Once the problem suite is established, contributors run benchmarks on their
machines and submit results. Results are stored as JSONL files for easy
comparison across machines and solver versions.

Getting started
---------------

.. code-block:: bash

   git clone https://github.com/cvxpy/solver-benchmarks.git
   cd solver-benchmarks
   uv sync
   uv run python run_benchmarks.py
   uv run python summarize.py

See the `GitHub repository <https://github.com/cvxpy/solver-benchmarks>`_ for
full instructions and contribution guidelines.
