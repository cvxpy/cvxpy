"""
Copyright 2022, the CVXPY developers.

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

import unittest

from cvxpy.utilities.versioning import Version


class TestVersioning(unittest.TestCase):

    def test_typical_inputs(self):
        self.assertTrue(Version('1.0.0') < Version('2.0.0'))
        self.assertTrue(Version('1.0.0') < Version('1.1.0'))
        self.assertTrue(Version('1.0.0') < Version('1.0.1'))
        self.assertFalse(Version('1.0.0') < Version('1.0.0'))
        self.assertTrue(Version('1.0.0') <= Version('1.0.0'))

        self.assertFalse(Version('1.0.0') >= Version('2.0.0'))
        self.assertFalse(Version('1.0.0') >= Version('1.1.0'))
        self.assertFalse(Version('1.0.0') >= Version('1.0.1'))
        self.assertTrue(Version('1.0.0') >= Version('1.0.0'))
        self.assertFalse(Version('1.0.0') > Version('1.0.0'))

    def test_tuple_construction(self):
        self.assertTrue(Version('0100.2.03') == Version((100, 2, 3)))
        self.assertTrue(Version('1.2.3') == Version((1, 2, 3, None)))
        self.assertTrue(Version('1.2.3') == Version((1, 2, 3, 'junk')))
        self.assertTrue(Version('1.2.3') == Version((1, 2, 3, -1)))

    def test_local_version_identifiers(self):
        self.assertTrue(Version('1.0.0') == Version('1.0.0+1'))
        self.assertTrue(Version('1.0.0') == Version('1.0.0+xxx'))
        self.assertTrue(Version('1.0.0') == Version('1.0.0+x.y.z'))
