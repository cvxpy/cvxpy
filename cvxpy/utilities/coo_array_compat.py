"""
Copyright 2025, the CVXPY authors.

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

# This file implements get_coords to paper over
# SciPy API changes between 1.12 and 1.13


def get_coords(c):
    try:
        return c.coords
    except AttributeError:
        return (c.row, c.col)
