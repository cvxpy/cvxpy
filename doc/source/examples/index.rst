.. _examples:

Examples
========

These examples show many different ways to use CVXPY.

* The :ref:`basic` section shows how to solve some common optimization problems
  in CVXPY.
* The :ref:`machine-learning` section is a tutorial covering convex methods in
  machine learning.
* The :ref:`advanced-python` and :ref:`applications` sections contains
  more complex examples aimed at experts in convex optimization.
* The :ref:`dgp-examples` section contains an interactive tutorial on :ref:`disciplined
  geometric programming <dgp>` and various examples of DGP problems.

.. _basic:

Basic Examples
--------------

- :doc:`Least-squares <basic/least_squares>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/WWW/least_squares.ipynb>`_

- `Control <http://nbviewer.ipython.org/github/cvxgrp/cvx_short_course/blob/master/intro/control.ipynb>`_

- `Portfolio optimization <http://nbviewer.ipython.org/github/cvxgrp/cvx_short_course/blob/master/applications/portfolio_optimization.ipynb>`_

- `Worst-case risk analysis <http://nbviewer.ipython.org/github/cvxgrp/cvx_short_course/blob/master/applications/worst_case_analysis.ipynb>`_

- `Model fitting <http://nbviewer.ipython.org/github/cvxgrp/cvx_short_course/blob/master/applications/model_fitting.ipynb>`_

- `Optimal advertising <http://nbviewer.ipython.org/github/cvxgrp/cvx_short_course/blob/master/applications/optimal_ad.ipynb>`_

- :doc:`Total variation in-painting <applications/tv_inpainting>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/WWW/tv_inpainting.ipynb>`_


.. _machine-learning:

Machine Learning
----------------

- :doc:`Ridge regression <machine_learning/ridge_regression>` `\[.py\] <http://github.com/cvxgrp/cvxpy/blob/1.0/examples/machine_learning/ridge_regression.py>`_ `\[.ipynb\] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/1.0/examples/machine_learning/ridge_regression.ipynb>`_

- :doc:`Lasso regression <machine_learning/lasso_regression>` `\[.py\] <http://github.com/cvxgrp/cvxpy/blob/1.0/examples/machine_learning/lasso_regression.py>`_ `\[.ipynb\] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/1.0/examples/machine_learning/lasso_regression.ipynb>`_

- `SVM classifier with regularization <http://nbviewer.ipython.org/github/cvxgrp/cvx_short_course/blob/master/intro/SVM.ipynb>`_

- `Huber regression <http://nbviewer.ipython.org/github/cvxgrp/cvx_short_course/blob/master/applications/huber_regression.ipynb>`_

- `Quantile regression <http://nbviewer.ipython.org/github/cvxgrp/cvx_short_course/blob/master/applications/quantile_regression.ipynb>`_

.. _advanced-python:

Advanced
--------

- :doc:`Object-oriented convex optimization <applications/OOCO>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/WWW/OOCO.ipynb>`_

- :doc:`Consensus optimization <applications/consensus_opt>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/WWW/consensus_opt.ipynb>`_

- :doc:`Method of multipliers <applications/MM>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/WWW/MM.ipynb>`_

.. _applications:

Advanced Applications
---------------------

- :doc:`Allocating interdiction effort to catch a smuggler <applications/interdiction>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/WWW/interdiction.ipynb>`_
- :doc:`Antenna array design <applications/ant_array_min_beamwidth>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/WWW/ant_array_min_beamwidth.ipynb>`_
- :doc:`Channel capacity <applications/Channel_capacity_BV4.57>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/WWW/Channel_capacity_BV4.57.ipynb>`_
- :doc:`Computing a sparse solution of a set of linear inequalities <applications/sparse_solution>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/WWW/sparse_solution.ipynb>`_
- :doc:`Entropy maximization <applications/max_entropy>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/WWW/max_entropy.ipynb>`_
- :doc:`Fault detection <applications/fault_detection>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/WWW/fault_detection.ipynb>`_
- :doc:`Filter design <applications/fir_chebychev_design>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/WWW/fir_chebychev_design.ipynb>`_
- :doc:`Fitting censored data <applications/censored_data>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/WWW/censored_data.ipynb>`_
- :doc:`L1 trend filtering <applications/l1_trend_filter>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/WWW/l1_trend_filter.ipynb>`_
- :doc:`Nonnegative matrix factorization <applications/nonneg_matrix_fact>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/WWW/nonneg_matrix_fact.ipynb>`_
- :doc:`Optimal parade route <applications/parade_route>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/WWW/parade_route.ipynb>`_
- :doc:`Optimal power and bandwidth allocation in a Gaussian broadcast channel <applications/optimal_power_gaussian_channel_BV4.62>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/WWW/optimal_power_gaussian_channel_BV4.62.ipynb>`_
- :doc:`Power assignment in a wireless communication system <applications/maximise_minimum_SINR_BV4.20>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/WWW/maximise_minimum_SINR_BV4.20.ipynb>`_
- :doc:`Predicting NBA game wins <applications/nba_ranking>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/WWW/nba_ranking.ipynb>`_
- :doc:`Robust Kalman filtering for vehicle tracking <applications/robust_kalman>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/WWW/robust_kalman.ipynb>`_
- :doc:`Sizing of clock meshes <applications/clock_mesh>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/WWW/clock_mesh.ipynb>`_
- :doc:`Sparse covariance estimation for Gaussian variables <applications/sparse_covariance_est>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/WWW/sparse_covariance_est.ipynb>`_
- :doc:`Water filling <applications/water_filling_BVex5.2>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/WWW/water_filling_BVex5.2.ipynb>`_

.. _dgp-examples:

Disciplined Geometric Programming
---------------------------------------
- :doc:`DGP fundamentals <dgp/dgp_fundamentals>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/dgp/dgp_fundamentals.ipynb>`_
- :doc:`Maximizing the volume of a box <dgp/max_volume_box>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/dgp/max_volume_box.ipynb>`_
- :doc:`Power control <dgp/power_control>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/dgp/power_control.ipynb>`_
- :doc:`Perron-Frobenius matrix completion <dgp/pf_matrix_completion>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/dgp/pf_matrix_completion.ipynb>`_
- :doc:`Rank-one nonnegative matrix factorization <dgp/rank_one_nmf>` `[.ipynb] <http://nbviewer.ipython.org/github/cvxgrp/cvxpy/blob/master/examples/notebooks/dgp/rank_one_nmf.ipynb>`_
