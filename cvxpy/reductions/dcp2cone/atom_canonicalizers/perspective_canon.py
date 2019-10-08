"""
Copyright 2019 Shane Barratt

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
from cvxpy.reductions.dcp2cone.atom_canonicalizers import CANON_METHODS

def perspective_canon(expr, args):
    # perspective(f)(x, t) = {
    #    tf(x/t)  if t > 0,
    #    0        if t = 0, x = 0
    #    infinity otherwise
    # } 
    x = args[:-1]
    y = args[-1].flatten()
    t = Variable(1,)

    t_underlying, constraints_underlying = CANON_METHODS[expr](*x)

    return t, constraints
