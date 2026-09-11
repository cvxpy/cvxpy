
Modelling chemical equilibrium
==============================

In this example, we compute the equilibrium composition of a gas mixture
by minimizing its Gibbs free energy. The instance is due to White,
Johnson, and Dantzig 1958, and considers the reaction products of a
mixture of :math:`(1/2)\mathrm{N}_2\mathrm{H}_4 + (1/2)\mathrm{O}_2`
at 3500 K and 750 psi.

There are :math:`n = 10` possible compounds,

.. math::

   \mathrm{H},\ \mathrm{H}_2,\ \mathrm{H}_2\mathrm{O},\ \mathrm{N},\
   \mathrm{N}_2,\ \mathrm{NH},\ \mathrm{NO},\ \mathrm{O},\
   \mathrm{O}_2,\ \mathrm{OH},

and :math:`m = 3` conserved elements, hydrogen, nitrogen, and oxygen.
Let :math:`x_j \geq 0` denote the number of moles of compound
:math:`j`, and let :math:`x_{\mathrm{tot}} = \sum_{j=1}^n x_j`.
The element balance constraints are

.. math::

   Ax = b,

where :math:`A_{ij}` is the number of atoms of element :math:`i` in one
molecule of compound :math:`j`. For this instance,
:math:`b = (2, 1, 1)`, corresponding to the total numbers of H, N, and O
atoms in the initial mixture after scaling.

At fixed temperature and pressure, the dimensionless Gibbs free energy is

.. math::

   \sum_{j=1}^n x_j \left(c_j + \log \frac{x_j}{x_{\mathrm{tot}}}\right),

where :math:`c_j = (F_j^0 / RT) + \log P` is a known coefficient for
compound :math:`j`. The equilibrium problem is therefore

.. math::

   \begin{array}{ll}
   \mbox{minimize} & \displaystyle
       \sum_{j=1}^n x_j \left(c_j + \log \frac{x_j}{x_{\mathrm{tot}}}\right) \\
   \mbox{subject to} & Ax = b \\
                     & x \geq 0,
   \end{array}

with variable :math:`x \in \mathbf{R}^n`. In CVXPY, the nonlinear term
can be written using the ``entr`` atom as

.. math::

   \sum_{j=1}^n x_j \log x_j - x_{\mathrm{tot}}\log x_{\mathrm{tot}}
   =
   -\sum_{j=1}^n \mathrm{entr}(x_j)
   + \mathrm{entr}(x_{\mathrm{tot}}).

The term :math:`\mathrm{entr}(x_{\mathrm{tot}})` is concave, so the
formulation is not DCP. It can nevertheless be passed to an NLP solver
through CVXPY's DNLP interface.

.. code:: python

   import cvxpy as cp
   import numpy as np

   compounds = ["H", "H2", "H2O", "N", "N2", "NH", "NO", "O", "O2", "OH"]
   n_compounds = len(compounds)

   # c_j = (F0/RT)_j + ln(P), where P = 750 psi
   c = np.array([
       -6.089,
       -17.164,
       -34.054,
       -5.914,
       -24.721,
       -14.986,
       -24.100,
       -10.708,
       -26.662,
       -22.179,
   ])

   # Stoichiometry matrix A[i,j] = atoms of element i in compound j
   # Rows: H, N, O
   A = np.array([
       [1, 2, 2, 0, 0, 1, 0, 0, 0, 1],
       [0, 0, 0, 1, 2, 1, 1, 0, 0, 0],
       [0, 0, 1, 0, 0, 0, 1, 1, 2, 1],
   ], dtype=float)

   b = np.array([2.0, 1.0, 1.0])

   x = cp.Variable(n_compounds, nonneg=True)
   x_total = cp.sum(x)

   gibbs_energy = c @ x - cp.sum(cp.entr(x)) + cp.entr(x_total)
   constraints = [A @ x == b]

   prob = cp.Problem(cp.Minimize(gibbs_energy), constraints)
   prob.solve(solver=cp.IPOPT, nlp=True, verbose=False)

   print("Optimal value:", prob.value)
   print("Equilibrium composition:")
   for compound, value in zip(compounds, x.value):
       print(f"{compound}: {value:.4f} moles")

.. parsed-literal::
   Optimal value: -47.761090859365865
   H: 0.0407 moles
   H2: 0.1477 moles
   H2O: 0.7832 moles
   N: 0.0014 moles
   N2: 0.4852 moles
   NH: 0.0007 moles
   NO: 0.0274 moles
   O: 0.0179 moles
   O2: 0.0373 moles
   OH: 0.0969 moles