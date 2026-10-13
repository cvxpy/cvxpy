"""
cvxpy.nlp — Namespace for NLP (nonlinear programming) atoms.

These atoms require a solver that supports nlp=True (e.g. IPOPT, UNO).

Example usage:
    import cvxpy as cp
    x = cp.Variable()
    prob = cp.Problem(cp.Minimize(cp.nlp.sin(x)), [x >= 0])
    prob.solve(nlp=True)
"""

from cvxpy.atoms.elementwise.hyperbolic import sinh, tanh, asinh, atanh
from cvxpy.atoms.elementwise.normcdf import normcdf
from cvxpy.atoms.elementwise.trig import atan, atan2, cos, sin, tan

__all__ = [
    "sin", "cos", "tan", "atan", "atan2", "normcdf",
    "sinh", "tanh", "asinh", "atanh",
]
