"""Tests for cp.nlp namespace."""
import cvxpy as cp
from cvxpy.atoms.elementwise.hyperbolic import tanh
from cvxpy.atoms.elementwise.trig import cos, sin


class TestNLPNamespace:
    def test_nlp_namespace_accessible(self):
        """Test that cp.nlp submodule is accessible."""
        assert hasattr(cp, 'nlp')

    def test_trig_atoms(self):
        x = cp.Variable()
        assert cp.nlp.sin(x) is not None
        assert cp.nlp.cos(x) is not None
        assert cp.nlp.tan(x) is not None

    def test_hyperbolic_atoms(self):
        x = cp.Variable()
        assert cp.nlp.sinh(x) is not None
        assert cp.nlp.tanh(x) is not None
        assert cp.nlp.asinh(x) is not None
        assert cp.nlp.atanh(x) is not None

    def test_atoms_not_in_top_level(self):
        """NLP atoms should only be accessible via cp.nlp, not cp directly."""
        assert not hasattr(cp, 'sin')
        assert not hasattr(cp, 'cos')
        assert not hasattr(cp, 'tan')
        assert not hasattr(cp, 'sinh')
        assert not hasattr(cp, 'tanh')
        assert not hasattr(cp, 'asinh')
        assert not hasattr(cp, 'atanh')

    def test_atoms_same_class_as_direct_import(self):
        """cp.nlp.sin should be the same class as a direct import."""
        assert cp.nlp.sin is sin
        assert cp.nlp.cos is cos
        assert cp.nlp.tanh is tanh
