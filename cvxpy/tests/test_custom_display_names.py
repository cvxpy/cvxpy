
"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import pytest

import cvxpy as cp

"""
Tests for custom display names on expressions and constraints.

This feature allows users to assign custom labels to expressions and constraints
for better debugging and dual variable interpretation via set_display_name() and
display_name() methods.
"""


class TestExpressionNaming:
    """Test naming functionality for expressions."""
    
    def test_expression_set_display_name_basic(self):
        """Test basic set_display_name functionality on expressions."""
        x = cp.Variable(3, name="x")
        expr = cp.sum(x)
        
        # Before setting display name
        assert not hasattr(expr, '_display_name')
        assert expr.display_name() == str(expr)
        assert expr.display_name() == expr.name()
        
        # After setting display name
        labeled_expr = expr.set_display_name("total_x")
        assert labeled_expr is expr  # Returns self for chaining
        assert hasattr(expr, '_display_name')
        assert expr._display_name == "total_x"
        assert expr.display_name() == "total_x: Sum(x, None, False)"
        
    def test_expression_str_unchanged(self):
        """Test that str() on expressions returns pure mathematical form."""
        x = cp.Variable(3, name="x")
        expr = cp.sum(x).set_display_name("labeled_sum")
        
        # str() should return pure math (unchanged behavior)
        assert str(expr) == "Sum(x, None, False)"
        assert expr.name() == "Sum(x, None, False)"
        assert expr.display_name() == "labeled_sum: Sum(x, None, False)"
        
    def test_expression_method_chaining(self):
        """Test method chaining with set_display_name."""
        x = cp.Variable(3, name="x")
        A = np.array([[1, 2, 3], [4, 5, 6]])
        
        expr = cp.sum_squares(A @ x).set_display_name("objective")
        assert expr._display_name == "objective"
        assert "objective:" in expr.display_name()
        
    def test_expression_display_name_none_handling(self):
        """Test setting display name to None."""
        x = cp.Variable(3, name="x")
        expr = cp.sum(x).set_display_name("test")
        
        # Clear the display name
        expr.set_display_name(None)
        assert not hasattr(expr, '_display_name') or expr._display_name is None
        assert expr.display_name() == expr.name()
        
        
    def test_compound_expression_no_label_contamination(self):
        """Test that labels don't leak into compound expressions."""
        x = cp.Variable(3, name="x")
        
        # Create labeled sub-expression
        base = cp.sum_squares(x).set_display_name("base_term")
        
        # Use it in compound expression
        compound = base + cp.norm(x)
        
        # The compound expression should have pure math in name() and str()
        compound_str = str(compound)
        compound_name = compound.name()
        
        # Should not contain "base_term:" in the mathematical representation
        assert "base_term:" not in compound_str
        assert "base_term:" not in compound_name
        
        # But base itself should still show its label in display_name()
        assert "base_term:" in base.display_name()


class TestConstraintNaming:
    """Test naming functionality for constraints."""
    
    def test_constraint_set_display_name_basic(self):
        """Test basic set_display_name functionality on constraints."""
        x = cp.Variable(3, name="x")
        con = x >= 0
        
        # Before setting display name
        assert not hasattr(con, '_display_name')
        assert con.display_name() == str(con)
        assert con.display_name() == con.name()
        
        # After setting display name
        labeled_con = con.set_display_name("non_negative")
        assert labeled_con is con  # Returns self for chaining
        assert hasattr(con, '_display_name')
        assert con._display_name == "non_negative"
        assert con.display_name() == "non_negative: 0.0 <= x"
        
    def test_constraint_str_shows_label(self):
        """Test that str() on constraints shows custom display names."""
        x = cp.Variable(3, name="x")
        con = (x >= 0).set_display_name("bounds")
        
        # str() should show label (convenient for constraints)
        assert str(con) == "bounds: 0.0 <= x"
        assert con.name() == "0.0 <= x"  # Pure math
        assert con.display_name() == "bounds: 0.0 <= x"  # Labeled
        
    def test_constraint_types_all_supported(self):
        """Test that all constraint types support naming."""
        x = cp.Variable(3, name="x")
        X = cp.Variable((2, 2), symmetric=True)
        
        # Inequality constraint
        ineq = (x >= 0).set_display_name("ineq_test")
        assert "ineq_test:" in str(ineq)
        
        # Equality constraint  
        eq = (cp.sum(x) == 1).set_display_name("eq_test")
        assert "eq_test:" in str(eq)
        
        # PSD constraint
        psd = (X >> 0).set_display_name("psd_test")
        assert "psd_test:" in str(psd)
        
        # Zero constraint (x == 0)
        zero = (x[0] == 0).set_display_name("zero_test")
        assert "zero_test:" in str(zero)
        
    def test_constraint_method_chaining(self):
        """Test method chaining with constraint set_display_name."""
        x = cp.Variable(3, name="x")
        
        con = (x >= 0).set_display_name("chaining_test")
        assert con._display_name == "chaining_test"
        assert "chaining_test:" in str(con)


class TestStringBehaviorDifferences:
    """Test that str() behaves differently for expressions vs constraints.
    
    Expressions: str() returns pure math to prevent display name contamination 
    in compound expressions. When CVXPY builds compound expressions (e.g., 
    expr1 + expr2), it calls str() on the sub-expressions to construct the 
    mathematical representation. If str() returned display names, those names 
    would leak into the math: "name1: x + name2: y" instead of "x + y".
    
    Constraints: str() returns display name form for user convenience since 
    constraints don't compose into compound forms.
    """
    
    def test_expression_str_shows_pure_math(self):
        """Test that str() on expressions shows pure mathematical form only."""
        x = cp.Variable(3, name="x")
        expr = cp.sum(x).set_display_name("test_expr")
        
        # API contract for expressions
        pure_math = expr.name()
        labeled = expr.display_name()
        str_result = str(expr)
        
        # str() and name() should be identical (pure math)
        assert str_result == pure_math
        assert "test_expr:" not in str_result
        assert "test_expr:" not in pure_math
        
        # display_name() should include label
        assert "test_expr:" in labeled
        assert pure_math in labeled  # Should contain the math part
        
    def test_constraint_str_shows_display_names(self):
        """Test that str() on constraints shows custom display names."""
        x = cp.Variable(3, name="x")
        con = (x >= 0).set_display_name("test_con")
        
        # API contract for constraints
        pure_math = con.name()
        labeled = con.display_name()
        str_result = str(con)
        
        # str() and display_name() should be identical (labeled)
        assert str_result == labeled
        assert "test_con:" in str_result
        assert "test_con:" in labeled
        
        # name() should be pure math
        assert "test_con:" not in pure_math
        assert pure_math in labeled  # Should contain the math part
        
    def test_unlabeled_behavior_unchanged(self):
        """Test that unlabeled expressions/constraints behave as before."""
        x = cp.Variable(3, name="x")
        expr = cp.sum(x)
        con = x >= 0
        
        # Should behave exactly as before labeling feature
        assert str(expr) == expr.name() == expr.display_name()
        assert str(con) == con.name() == con.display_name()


class TestPracticalUseCases:
    """Test practical use cases from GitHub issues."""
    
    def test_dual_variable_debugging(self):
        """Test the dual variable debugging use case."""
        x = cp.Variable(3, name="x")
        
        constraints = [
            (x >= 0).set_display_name("non_negative_weights"),
            (x <= 1).set_display_name("upper_bounds"), 
            (cp.sum(x) == 1).set_display_name("budget_constraint")
        ]
        
        # Should be easy to identify constraints by name
        for con in constraints:
            # This would be useful for dual value debugging
            con_label = str(con)  # Shows label automatically
            
            assert ":" in con_label  # Has label
            assert con.name() in con_label  # Contains math
            
    def test_complex_optimization_model(self):
        """Test naming in a realistic optimization model."""
        n_assets = 5
        weights = cp.Variable(n_assets, name="weights")
        returns = np.random.randn(n_assets)
        cov_matrix = np.eye(n_assets)  # Simple covariance
        
        # Expressions with business meaning
        portfolio_return = (returns.T @ weights).set_display_name("expected_return")
        portfolio_risk = cp.quad_form(weights, cov_matrix).set_display_name("portfolio_variance")
        
        # Constraints with business meaning
        constraints = [
            (weights >= 0).set_display_name("long_only"),
            (cp.sum(weights) == 1).set_display_name("fully_invested"),
            (weights <= 0.4).set_display_name("concentration_limits")
        ]
        
        # Check that expressions maintain pure math in str()
        assert "expected_return:" not in str(portfolio_return)
        assert "portfolio_variance:" not in str(portfolio_risk)
        
        # But labels are available via display_name()
        assert "expected_return:" in portfolio_return.display_name()
        assert "portfolio_variance:" in portfolio_risk.display_name()
        
        # Constraints show labels in str() for convenience
        for con in constraints:
            assert ":" in str(con)
            assert con._display_name in str(con)
            
    def test_error_messages_readability(self):
        """Test that labeled constraints improve error message readability."""
        x = cp.Variable(3, name="x")
        
        # Create constraints that might appear in error messages
        constraints = [
            (x >= 0).set_display_name("box_constraint_lower"),
            (x <= 10).set_display_name("box_constraint_upper")
        ]
        
        # When these appear in error messages, they should be readable
        for con in constraints:
            error_repr = str(con)  # What would appear in error
            assert con._display_name in error_repr
            assert ":" in error_repr
            # The mathematical form should still be present
            assert any(op in error_repr for op in [">=", "<=", "=="])


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_display_name(self):
        """Test edge cases with empty or whitespace display names."""
        x = cp.Variable(3, name="x")
        expr = cp.sum(x)
        
        # Empty string
        expr.set_display_name("")
        assert expr._display_name == ""
        assert expr.display_name() == ": Sum(x, None, False)"
        
        # Whitespace
        expr.set_display_name("   ")
        assert expr._display_name == "   "
        assert expr.display_name() == "   : Sum(x, None, False)"
        
    def test_unicode_display_names(self):
        """Test display names with unicode characters."""
        x = cp.Variable(3, name="x")
        expr = cp.sum(x).set_display_name("λ_objective")
        
        assert "λ_objective:" in expr.display_name()
        
    def test_very_long_display_names(self):
        """Test with very long display names."""
        x = cp.Variable(3, name="x")
        long_name = "a" * 1000
        expr = cp.sum(x).set_display_name(long_name)
        
        assert expr._display_name == long_name
        assert long_name in expr.display_name()


if __name__ == "__main__":
    pytest.main([__file__])