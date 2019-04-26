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
from cvxpy.atoms import ceil, floor


def ceil_inv(t):
    return floor(t)


def floor_inv(t):
    return ceil(t)


INVERSES = {
    ceil: ceil_inv,
    floor: floor_inv,
}


INVERTIBLE = set(INVERSES.keys())


def invertible(expr):
    return type(expr) in INVERTIBLE


def inverse(expr):
    return INVERSES[type(expr)]
