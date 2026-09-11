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

from cvxpy.tests.base_test import BaseTest
from cvxpy.utilities import scopes


class TestScopes(BaseTest):
    """
    Tests for the scope managers.
    """

    def get_state(self, name):
        return getattr(scopes._thread_local, name, False)

    def test_dpp_scope(self):
        with scopes.dpp_scope():
            assert self.get_state("dpp_scope_active")

        assert not self.get_state("dpp_scope_active")

    def test_dpp_scope_exception(self):
        with self.assertRaises(AssertionError):
            with scopes.dpp_scope():
                raise AssertionError

        assert not self.get_state("dpp_scope_active")

    def test_dpp_scope_active(self):
        with scopes.dpp_scope():
            assert scopes.dpp_scope_active()
        assert not scopes.dpp_scope_active()

    def test_quad_form_dpp_scope(self):
        with scopes.quad_form_dpp_scope():
            assert self.get_state("quad_form_dpp_scope_active")

        assert not self.get_state("quad_form_dpp_scope_active")

    def test_quad_form_dpp_scope_exception(self):
        with self.assertRaises(AssertionError):
            with scopes.quad_form_dpp_scope():
                raise AssertionError

        assert not self.get_state("quad_form_dpp_scope_active")

    def test_quad_form_dpp_scope_active(self):
        with scopes.quad_form_dpp_scope():
            assert scopes.quad_form_dpp_scope_active()
        assert not scopes.quad_form_dpp_scope_active()
