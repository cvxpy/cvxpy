"""
Copyright 2025 Steven Diamond

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

from cvxpy.reductions.solvers.solver_inverse_data import SolverInverseData


class TestSolverInverseData:
    """Tests for the SolverInverseData class."""

    def test_get_method_with_dict(self):
        """Test get() method with dictionary inverse_data (typical solver use case)."""
        # This mimics how solvers return inv_data as a dict from apply()
        inv_data_dict = {
            "is_mip": True,
            "var_id": 123,
            "offset": 5.0
        }
        solver_inv_data = SolverInverseData(inv_data_dict)

        # Test get() with existing keys
        assert solver_inv_data.get("is_mip", False)
        assert solver_inv_data.get("var_id", 0) == 123
        assert solver_inv_data.get("offset", 0) == 5.0

        # Test get() with missing key returns default
        assert not solver_inv_data.get("missing_key", False)
        assert solver_inv_data.get("another_missing", "default") == "default"

        # Test __contains__
        assert "is_mip" in solver_inv_data
        assert "missing_key" not in solver_inv_data
