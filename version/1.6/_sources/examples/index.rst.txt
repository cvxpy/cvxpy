.. _examples:

Examples
========

These examples show many different ways to use CVXPY.

* The :ref:`basic` section shows how to solve some common optimization problems
  in CVXPY.
* The :ref:`dgp-examples` section shows how to solve log-log convex programs.
* The :ref:`dqcp-examples` section has examples on quasiconvex programming.
* The :ref:`derivative-examples` section shows how to compute sensitivity analyses and gradients of solutions.

There are also application-specific sections.

* The :ref:`machine-learning` section is a tutorial on convex optimization in
  machine learning.
* The :ref:`advanced-python` and :ref:`applications` sections contains
  more complex examples for experts in convex optimization.
  
.. _basic:

Basic examples
--------------

- :doc:`Least squares <basic/least_squares>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/least_squares.ipynb>`_

- :doc:`Linear program <basic/linear_program>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/linear_program.ipynb>`_

- :doc:`Quadratic program <basic/quadratic_program>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/quadratic_program.ipynb>`_

- :doc:`Second-order cone program <basic/socp>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/socp.ipynb>`_

- :doc:`Semidefinite program <basic/sdp>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/sdp.ipynb>`_

- :doc:`Mixed-integer quadratic program <basic/mixed_integer_quadratic_program>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/mixed_integer_quadratic_program.ipynb>`_

- `Control <https://colab.research.google.com/github/cvxgrp/cvx_short_course/blob/master/book/docs/intro/notebooks/control.ipynb>`_

- `Portfolio optimization <https://colab.research.google.com/github/cvxgrp/cvx_short_course/blob/master/book/docs/applications/notebooks/portfolio_optimization.ipynb>`_

- `Worst-case risk analysis <https://colab.research.google.com/github/cvxgrp/cvx_short_course/blob/master/book/docs/applications/notebooks/worst_case_analysis.ipynb>`_

- `Model fitting <https://colab.research.google.com/github/cvxgrp/cvx_short_course/blob/master/book/docs/applications/notebooks/model_fitting.ipynb>`_

- `Optimal advertising <https://colab.research.google.com/github/cvxgrp/cvx_short_course/blob/master/book/docs/applications/notebooks/optimal_ad.ipynb>`_

- :doc:`Total variation in-painting <applications/tv_inpainting>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/tv_inpainting.ipynb>`_


.. _dgp-examples:

Disciplined geometric programming
---------------------------------------
- :doc:`DGP fundamentals <dgp/dgp_fundamentals>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/dgp/dgp_fundamentals.ipynb>`_
- :doc:`Maximizing the volume of a box <dgp/max_volume_box>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/dgp/max_volume_box.ipynb>`_
- :doc:`Power control <dgp/power_control>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/dgp/power_control.ipynb>`_
- :doc:`Perron-Frobenius matrix completion <dgp/pf_matrix_completion>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/dgp/pf_matrix_completion.ipynb>`_
- :doc:`Rank-one nonnegative matrix factorization <dgp/rank_one_nmf>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/dgp/rank_one_nmf.ipynb>`_


.. _dqcp-examples:

Disciplined quasiconvex programming
-----------------------------------
- :doc:`Concave fractional function <dqcp/concave_fractional_function>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/dqcp/concave_fractional_function.ipynb>`_
- :doc:`Minimum-length least squares <dqcp/minimum_length_least_squares>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/dqcp/minimum_length_least_squares.ipynb>`_
- :doc:`Hypersonic shape design <dqcp/hypersonic_shape_design>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/dqcp/hypersonic_shape_design.ipynb>`_


.. _derivative-examples:

Derivatives
-----------
- :doc:`Fundamentals <derivatives/fundamentals>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/derivatives/fundamentals.ipynb>`_
- :doc:`Queuing design <derivatives/queuing_design>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/derivatives/queuing_design.ipynb>`_
- :doc:`Structured prediction <derivatives/structured_prediction>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/derivatives/structured_prediction.ipynb>`_

.. _machine-learning:

Machine learning
----------------

- :doc:`Ridge regression <machine_learning/ridge_regression>` `\[.ipynb\] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/machine_learning/ridge_regression.ipynb>`_

- :doc:`Lasso regression <machine_learning/lasso_regression>` `\[.ipynb\] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/machine_learning/lasso_regression.ipynb>`_

- :doc:`Logistic regression <machine_learning/logistic_regression>` `\[.ipynb\] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/machine_learning/logistic_regression.ipynb>`_

- :doc:`SVM classifier <machine_learning/svm>` `\[.ipynb\] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/machine_learning/svm.ipynb>`_

- `Huber regression <https://colab.research.google.com/github/cvxgrp/cvx_short_course/blob/master/book/docs/applications/notebooks/huber_regression.ipynb>`_

- `Quantile regression <https://colab.research.google.com/github/cvxgrp/cvx_short_course/blob/master/book/docs/applications/notebooks/quantile_regression.ipynb>`_

.. _finance

Finance
-------

- `Portfolio optimization <https://colab.research.google.com/github/cvxgrp/cvx_short_course/blob/master/book/docs/applications/notebooks/portfolio_optimization.ipynb>`_

- `Cryptocurrency trading <https://nbviewer.org/github/rcroessmann/sharing_public/blob/master/arbitrage_identification.ipynb>`_

- `Entropic Portfolio Optimization <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/Entropic%20Portfolio.ipynb>`_

- `Portfolio Optimization using SOC constraints <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/SOC%20Portfolio.ipynb>`_

- `Gini Mean Difference Portfolio Optimization <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/Gini%20Portfolio.ipynb>`_

- `Kurtosis Portfolio Optimization <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/Kurtosis%20Portfolio.ipynb>`_

- `Relativistic Value at Risk Portfolio Optimization <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/Relativistic%20Value%20at%20Risk%20Portfolio.ipynb>`_

- `Approximate Kurtosis Portfolio Optimization <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/Approximate%20Kurtosis%20Portfolio.ipynb>`_

.. _advanced-python:

Advanced
--------

- :doc:`Object-oriented convex optimization <applications/OOCO>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/OOCO.ipynb>`_

- :doc:`Consensus optimization <applications/consensus_opt>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/consensus_opt.ipynb>`_

- :doc:`Method of multipliers <applications/MM>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/MM.ipynb>`_

.. _applications:

Advanced Applications
---------------------

- :doc:`Allocating interdiction effort to catch a smuggler <applications/interdiction>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/interdiction.ipynb>`_
- :doc:`Antenna array design <applications/ant_array_min_beamwidth>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/ant_array_min_beamwidth.ipynb>`_
- :doc:`Channel capacity <applications/Channel_capacity_BV4.57>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/Channel_capacity_BV4.57.ipynb>`_
- :doc:`Computing a sparse solution of a set of linear inequalities <applications/sparse_solution>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/sparse_solution.ipynb>`_
- :doc:`Entropy maximization <applications/max_entropy>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/max_entropy.ipynb>`_
- :doc:`Fault detection <applications/fault_detection>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/fault_detection.ipynb>`_
- :doc:`Filter design <applications/fir_chebychev_design>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/fir_chebychev_design.ipynb>`_
- :doc:`Fitting censored data <applications/censored_data>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/censored_data.ipynb>`_
- :doc:`L1 trend filtering <applications/l1_trend_filter>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/l1_trend_filter.ipynb>`_
- :doc:`Nonnegative matrix factorization <applications/nonneg_matrix_fact>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/nonneg_matrix_fact.ipynb>`_
- :doc:`Optimal parade route <applications/parade_route>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/parade_route.ipynb>`_
- :doc:`Optimal power and bandwidth allocation in a Gaussian broadcast channel <applications/optimal_power_gaussian_channel_BV4.62>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/optimal_power_gaussian_channel_BV4.62.ipynb>`_
- :doc:`Power assignment in a wireless communication system <applications/maximise_minimum_SINR_BV4.20>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/maximise_minimum_SINR_BV4.20.ipynb>`_
- :doc:`Predicting NBA game wins <applications/nba_ranking>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/nba_ranking.ipynb>`_
- :doc:`Robust Kalman filtering for vehicle tracking <applications/robust_kalman>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/robust_kalman.ipynb>`_
- :doc:`Sizing of clock meshes <applications/clock_mesh>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/clock_mesh.ipynb>`_
- :doc:`Sparse covariance estimation for Gaussian variables <applications/sparse_covariance_est>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/sparse_covariance_est.ipynb>`_
- :doc:`Water filling <applications/water_filling_BVex5.2>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/water_filling_BVex5.2.ipynb>`_
- `Multiple Traveling Salesman Problem <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/mTSP_en.ipynb>`_
- :doc:`Minimize Condition Number by Scaling <applications/min_condition_number_by_scaling>` `[.ipynb] <https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/min_condition_number_by_scaling.ipynb>`_
