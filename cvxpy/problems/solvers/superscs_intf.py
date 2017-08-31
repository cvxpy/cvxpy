"""
Copyright 2017 Steven Diamond

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

import cvxpy.settings as s
from cvxpy.problems.solvers.ecos_intf import ECOS
import numpy as np


class SUPERSCS(SCS):
    """An interface for the SuperSCS solver.
    """

    def name(self):
        """The name of the solver.
        """
        return s.SUPERSCS

    def import_solver(self):
        """Imports the solver.
        """
        import superscs
        superscs  # For flake8
