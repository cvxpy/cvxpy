"""Tests for cp.nlp namespace."""
import cvxpy as cp


class TestNLPNamespace:
    def test_nlp_namespace_accessible(self):
        """Test that cp.nlp submodule is accessible."""
        assert hasattr(cp, 'nlp')

    def test_trig_atoms(self):
        x = cp.Variable()
        assert str(cp.nlp.sin(x)) == 'sin(var1)'  
        assert str(cp.nlp.cos(x)) == 'cos(var1)'
        assert str(cp.nlp.tan(x)) == 'tan(var1)'

    def test_hyperbolic_atoms(self):
        x = cp.Variable()
        assert cp.nlp.sinh(x) is not None
        assert cp.nlp.tanh(x) is not None
        assert cp.nlp.asinh(x) is not None
        assert cp.nlp.atanh(x) is not None

    def test_atoms_same_as_direct(self):
        """cp.nlp.sin and cp.sin should be the same class."""
        assert cp.nlp.sin is cp.sin
        assert cp.nlp.cos is cp.cos
        assert cp.nlp.tanh is cp.tanh
