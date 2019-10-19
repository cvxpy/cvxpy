Expressions
==========================

CVXPY represents mathematical objects as *expression trees*. An expression
tree is a collection of mathematical expressions linked together by one or more
atoms. Expression trees are encoded as instances of the
:class:`~cvxpy.expressions.expression.Expression` class, and each
:class:`~cvxpy.expressions.leaf.Leaf` in a tree is a
:class:`~cvxpy.expressions.variable.Variable`,
:class:`~cvxpy.expressions.constants.parameter.Parameter`, or
:class:`~cvxpy.expressions.constants.constant.Constant`.

.. contents:: :local:


Expression
-------------------------------------

.. autoclass:: cvxpy.expressions.expression.Expression
    :members: value, grad, domain, name, curvature, is_constant, is_affine,
              is_convex, is_concave, is_dcp, is_log_log_affine,
              is_log_log_convex, is_log_log_concave, is_dgp, sign, is_zero,
              is_nonneg, is_nonpos, shape, size, ndim, T, __pow__, __add__,
              __radd__, __sub__, __rsub__, __mul__, __rmul__, __matmul__,
              __rmatmul__, __div__, __rdiv__, __rshift__, __rrshift__,
              __lshift__, __rlshift__, __eq__, __le__, __ge__, __truediv__,
              __rtruediv__
    :undoc-members:
    :show-inheritance:

Leaf
-------------------------------

.. autoclass:: cvxpy.expressions.leaf.Leaf
    :members: shape, size, ndim, T, value, project, project_and_assign
    :undoc-members:
    :show-inheritance:

Variable
-----------------------------------

.. autoclass:: cvxpy.expressions.variable.Variable
    :members: shape, size, ndim, T, value, project, project_and_assign, name
    :undoc-members:
    :show-inheritance:

Parameter
-----------------------------------

.. autoclass:: cvxpy.expressions.constants.parameter.Parameter
    :members: shape, size, ndim, T, value, project, project_and_assign, round
    :undoc-members:
    :show-inheritance:

Constant
-----------------------------------

.. autoclass:: cvxpy.expressions.constants.Constant
    :members: shape, size, ndim, T, value
    :undoc-members:
    :show-inheritance:
