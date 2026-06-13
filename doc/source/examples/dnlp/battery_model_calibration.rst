
Battery model calibration
=========================

*This example is adapted from Cederberg, Zhang, Nobel, and Boyd,* "`Disciplined Nonlinear Programming <https://stanford.edu/~boyd/papers/pdf/dnlp.pdf>`__\ ".

In this example, we use CVXPY to calibrate the unknown parameters of a
dynamic model of a lithium ion (Li-Ion) storage battery, based on
experiments. We consider the so-called Thévenin model, which models the battery as an open-circuit voltage source in
series with an internal resistance :math:`R_0`, in :math:`\Omega`
(Ohms), and a parallel resistance-capacitance (RC) pair
:math:`(R_1, C_1)`, with :math:`R_1` in :math:`\Omega` and :math:`C_1`
in F (Farads). We consider a time interval of :math:`T` seconds, and
denote the charging current at time :math:`t \in [0, T]` by
:math:`i(t)`, in A (Amperes). We let :math:`q(t)` denote the stored
charge, in C (Coulombs), :math:`v(t)` the terminal voltage, in V
(Volts), and :math:`v^{oc}(t)` the open-circuit voltage, in V. The
terminal voltage is

.. math::


   v(t) = v^{oc}(t) + R_0 i(t) + U^{RC}(t),

where :math:`U^{RC}(t)` is the voltage across the RC pair, which evolves
according to

.. math::


   \frac{dU^{RC}(t)}{dt} = -\frac{U^{RC}(t)}{R_1 C_1} + \frac{i(t)}{C_1}.

The stored charge :math:`q(t)` satisfies :math:`dq(t)/dt = i(t)`. We
model the open-circuit voltage as a function of the stored charge, as

.. math::


   v^{oc}(t) = a + \frac{b}{Q^{\mathrm{crit}} - q(t)},

where :math:`a`, :math:`b`, and :math:`Q^{\mathrm{crit}}` are model
parameters. The unit of :math:`a` is V, the unit of :math:`b` is J
(Joules), and :math:`Q^{\mathrm{crit}}` is given in C. The stored
battery charge :math:`q(t)` is always less than the critical charge
:math:`Q^{\mathrm{crit}}`. This model is parameterized by the six
positive parameters

.. math::


   a, \quad b, \quad Q^{\mathrm{crit}}, \quad R_0, \quad R_1, \quad C_1.
