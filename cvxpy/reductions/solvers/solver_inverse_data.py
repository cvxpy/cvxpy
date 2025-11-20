"""
Copyright 2013 Steven Diamond, 2017 Akshay Agrawal

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

from typing import Union

from cvxpy.reductions.inverse_data import InverseData


class SolverInverseData(InverseData):
    """
    The context for a reduction, which may include a solver and solver options.
    """
    def __init__(self, inverse_data: Union[list, InverseData], solver_instance = None,
                 solver_options: dict = {}):
        self.solver_instance = solver_instance
        self.solver_options = solver_options
        self.inverse_data = inverse_data

    def __getitem__(self, key):
        return self.inverse_data[key]