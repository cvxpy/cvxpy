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


WARN = False


def disable_warnings() -> None:
    global WARN
    WARN = False


def enable_warnings() -> None:
    global WARN
    WARN = True


def warnings_enabled() -> bool:
    return WARN


class SolverError(Exception):
    """Error thrown when a solver encounters an error solving a problem.
    """


class DCPError(Exception):
    """Error thrown for DCP violations.
    """


class DPPError(Exception):
    """Error thrown for DPP violations.
    """


class DGPError(Exception):
    """Error thrown for DGP violations.
    """


class DQCPError(Exception):
    """Error thrown for DQCP violations.
    """


class ParameterError(Exception):
    """Error thrown for accessing the value of an unspecified parameter.
    """
