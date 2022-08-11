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

from dataclasses import dataclass
import cvxpy as cp
from cvxpy import Variable
from cvxpy.constraints.second_order import SOC
from cvxpy.constraints.nonpos import NonNeg
from cvxpy.constraints.constraint import Constraint
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.zero import Zero
from cvxpy.constraints.power import PowCone3D
from cvxpy.constraints.psd import PSD
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class PerspectiveRepresentation:
    constraints: List[cp.Expression]
    t: cp.Variable
    s: cp.Variable

def form_cone_constraint(z: Variable ,constraint: Constraint)-> PerspectiveRepresentation: 
    """
    Given a constraint represented as Ax+b \in K for K a cvxpy cone, return an
    instantiated cvxpy constraint.
    """
    if isinstance(constraint, SOC):
        # TODO: Figure out how to instantiate Ax+b \in SOC where we know which
        # lines from our ultimate A_pers(x,t,s) + b \in K \times ... correspond
        # to this constraint. 
        return SOC(t = z[0]
                ,X = z[1:])
    elif isinstance(constraint, NonNeg):
        return NonNeg(z)
    elif isinstance(constraint,ExpCone):
        n = z.shape[0]
        assert len(z.shape) == 1
        assert n % 3 == 0 # we think this is how the exponential cone works
        step = n//3
        return ExpCone(z[:step],z[step:-step],z[-step:])
    elif isinstance(constraint,Zero):
        return Zero(z)
    elif isinstance(constraint,PSD):
        assert len(z.shape) == 1 
        N = z.shape[0]
        n = int(N**.5)
        assert N == n**2, "argument is not a vectorized square matrix"
        z_mat = cp.reshape(z,(n,n))
        return PSD(z_mat) # do we need constraint_id?
    elif isinstance(constraint,PowCone3D):
        raise NotImplementedError
    else:
        raise NotImplementedError

def form_perspective_from_f_exp(f_exp: cp.Expression,args: List[cp.Expression]):
    # Only working for minimization right now. 

    aux_prob = cp.Problem(cp.Minimize(f_exp))
    # Does numerical solution value of epigraph t coincisde with f_exp numerical
    # value at opt?

    chain = aux_prob._construct_chain()
    chain.reductions = chain.reductions[:-1] #skip solver reduction
    prob_canon = chain.apply(aux_prob)[0] #grab problem instance
    # get cone representation of c, A, and b for some problem.
    
    c = prob_canon.c.toarray().flatten()[:-1] # TODO: why
    d = prob_canon.c.toarray().flatten()[-1]
    Ab = prob_canon.A.toarray().reshape((-1,len(c)+1),order="F")
    A, b = Ab[:,:-1], Ab[:,-1]

    cones = prob_canon.cone_dims # get the cones used and the dims

    # given f in epigraph form, aka epi f = \{(x,t) | f(x) \leq t\} 
    # = \{(x,t) | Fx +tg + e \in K} for K a cone, the epigraph of the
    # perspective, \{(x,s,t) | sf(x/s) \leq t} = \{(x,s,t) | Fx + tg + se \in K\}
    # If I have the problem "minimize f(x)" written in the CVXPY compatible 
    # "c^Tx, Ax+b \in K" form, I can re-write this in the graph form above via
    # x,t \in \epi f iff Ax + b \in K and t-c^Tx \in R_+ which I can further write
    # with block matrices as Fx + tg + e \in K \times R_+
    # with F = [A ], g = [0], e = [b]  
    #          [-c]      [1]      [-d]

    # Actually, all we need is Ax + 0*t + sb \in K, -c^Tx + t - ds >= 0

    t = cp.Variable()
    s = args[1]
    x_canon = prob_canon.x
    constraints = []

    x_pers = A@x_canon + s*b
    
    i = 0
    for con in prob_canon.constraints:
        sz, tp = con.size, type(con)
        guy = x_pers[i:i+sz]
        pers_constraint = form_cone_constraint(guy,con)
        constraints.append(pers_constraint)
        i += sz
    # constraints.append(s >= 0) # from construction s should be nonneg
    constraints.append(-c@x_canon + t - s*d >= 0)

    # recover initial variables

    for var in f_exp.variables():
        start_ind = prob_canon.var_id_to_col[var.id]
        end_ind = start_ind+var.size
        constraints.append(cp.vec(var) == x_canon[start_ind:end_ind])

    return PerspectiveRepresentation(constraints,t,s)