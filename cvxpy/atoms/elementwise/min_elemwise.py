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
from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.atoms.elementwise.max_elemwise import max_elemwise


def min_elemwise(*args):
    if len(args) == 0 or (len(args) == 1 and not isinstance(args[0], list)):
        raise TypeError("min_elemwise requires at least two arguments or a list.")
    elif len(args) == 1:
        args = args[0]
    return -max_elemwise([-Elementwise.cast_to_const(arg) for arg in args])
