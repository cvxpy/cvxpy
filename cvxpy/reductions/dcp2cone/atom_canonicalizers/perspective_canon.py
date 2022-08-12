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
from cvxpy.perspective.perspective_utils import form_cone_constraint

def perspective_canon(expr, args):
    # Only working for minimization right now. 

    aux_prob = cp.Problem(cp.Minimize(expr.f))
    # Does numerical solution value of epigraph t coincisde with expr.f numerical
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
    s = args[0]
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

    end_inds = sorted(prob_canon.var_id_to_col.values()) + [x_canon.shape[0]] 

    for var in expr.f.variables():
        start_ind = prob_canon.var_id_to_col[var.id]
        end_ind = end_inds[end_inds.index(start_ind)+1]
        if var.is_psd():
            # Handling this separately now, maybe there is a better way. A
            # symmetric n x n matrix only is represented in the canonicalized
            # form via n(n-1)/2 variables.
        
            n = var.shape[0]
            inds = np.triu_indices(n, k=0) # includes diagonal
            constraints += [var[inds] == x_canon[start_ind:end_ind]]

        else:    
            constraints.append(cp.vec(var) == x_canon[start_ind:end_ind])
    
    return t, constraints