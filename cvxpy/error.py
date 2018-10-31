"""
Copyright 2013 Steven Diamond

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


class SolverError(Exception):
    """Error thrown when a solver encounters an error solving a problem.
    """
    pass


class DCPError(Exception):
    """Error thrown for DCP violations.
    """
    pass


class DGPError(Exception):
    """Error thrown for DGP violations.
    """
    pass


class ParameterError(Exception):
    """Error thrown for accessing the value of an unspecified parameter.
    """
    pass
