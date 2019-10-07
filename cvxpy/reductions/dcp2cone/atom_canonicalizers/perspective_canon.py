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

def perspective_canon(expr, args):
    # perspective(f)(x, t) = {
    #    tf(x/t)  if t > 0,
    #    0        if t = 0, x = 0
    #    infinity otherwise
    # } 
    x = args[0]
    y = args[1].flatten()
    # precondition: shape == ()
    t = Variable(1,)
    # (y+t, y-t, 2*x) must lie in the second-order cone,
    # where y+t is the scalar part of the second-order
    # cone constraint.
    constraints = [SOC(
                       t=y+t,
                       X=hstack([y-t, 2*x.flatten()]), axis=0
                      )]
    return t, constraints
