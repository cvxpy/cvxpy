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


class ProblemData(object):
    """A wrapper for the symbolic and numerical data for a problem.

    Attributes
    ----------
    sym_data : SymData
        The symbolic data for the problem.
    matrix_data : MatrixData
        The numerical data for the problem.
    prev_result : dict
        The result of the last solve.
    """

    def __init__(self):
        self.sym_data = None
        self.matrix_data = None
        self.prev_result = None
