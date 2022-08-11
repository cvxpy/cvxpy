"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import cvxpy as cp
from cvxpy.perspective.perspective_utils import form_perspective_from_f_exp


def perspective_canon(expr, args):
    # TODO: make sure expr is canonical form before solver, and then use instead
    # of re-canonicalizing in form_perspective_from_f_exp, also move bulk of the
    # former to here. 

    persp = form_perspective_from_f_exp(expr.f,args)
    
    return persp.t, persp.constraints