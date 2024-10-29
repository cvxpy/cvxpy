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

import numpy as np
from scipy import sparse

import cvxpy as cp
from cvxpy import problems
from cvxpy.constraints.psd import PSD
from cvxpy.constraints.second_order import SOC
from cvxpy.reductions.reduction import Reduction


class SOC2PSD(Reduction):
    """Convert all SOC constraints to equivalent PSD constraints.
    """

    def accepts(self, problem):
        return True

    def apply(self, problem):
        soc_constraints = []
        other_constraints = []
        for constraint in problem.constraints:
            if type(constraint) is SOC:
                soc_constraints.append(constraint)
            else:
                other_constraints.append(constraint)

        psd_constraints = []

        soc_constraint_ids = []
        soc_id_from_psd = {}

        for constraint in soc_constraints:
            r"""
            The SOC constraint :math:`\lVert X \rVert_2 \leq t` is modeled by `t` and `X`.
            We extract these `t` and `X` from the SOC constraint object.
            """
            t, X = constraint.args
            soc_constraint_ids.append(constraint.id)

            """
            A PSD constraint object will constrain the matrix M specified in its constructor to
            the PSD cone.

            We will create this matrix M using the `t` and `X` extracted from the SOC constraint.

            Since M being PSD means its Schur complement is also PSD, replacing :math:`M >> 0`
            with :math:`SchurComplement(M) >> 0` should give us the original SOC constraint.
            """

            if t.shape==(1,): # when `constraint` object has only one constraint
                scalar_term = t[0]
                vector_term_len = X.shape[0]

                """
                We construct the terms A, B and C that comprise the Schur complement of M
                There are multiple ways to construct A, B and C (and hence M) that are equivalent
                however, this one makes writing `invert` routine simple.
                """

                A = scalar_term * sparse.eye(1)
                B = cp.reshape(X,[-1,1], order='F').T
                C = scalar_term * sparse.eye(vector_term_len)

                """
                Another technique for reference

                A = scalar_term * sparse.eye(vector_term_len)
                B = cp.reshape(X,[-1,1], order='F')
                C = scalar_term * sparse.eye(1)
                """

                """
                Construct M from A, B and C
                """
                M = cp.bmat([
                    [A, B],
                    [B.T, C]
                ])

                """
                Constrain M to the PSD cone.
                """
                new_psd_constraint = PSD(M)
                soc_id_from_psd[new_psd_constraint.id] = constraint.id
                psd_constraints.append(new_psd_constraint)
            else: # when `constraint` object has multiple packed constraints
                if constraint.axis==1:
                    X = X.T
                for subidx in range(t.shape[0]):
                    scalar_term = t[subidx]
                    vector_term_len = X.shape[0]

                    A = scalar_term * sparse.eye(1)
                    B = X[:,subidx:subidx+1].T
                    C = scalar_term * sparse.eye(vector_term_len)

                    M = cp.bmat([
                        [A, B],
                        [B.T, C]
                    ])

                    new_psd_constraint = PSD(M)
                    soc_id_from_psd[new_psd_constraint.id] = constraint.id
                    psd_constraints.append(new_psd_constraint)

        new_problem = problems.problem.Problem(problem.objective,
                                               other_constraints + psd_constraints)

        inverse_data = (soc_id_from_psd, soc_constraint_ids)
        return new_problem, inverse_data

    def invert(self, solution, inverse_data):
        """
        `solution.dual_vars` contains dual variables corresponding to the constraints.

        The dual variables that we return in `solution` should correspond to the original
        SOC constraints, and not their PSD equivalents. To this end, inversion is required.
        """
        if solution.dual_vars=={}:
            # in case the solver fails
            return solution

        soc_id_from_psd, soc_constraint_ids = inverse_data
        psd_constraint_ids = soc_id_from_psd.keys()

        inverted_dual_vars = {}
        for constr_id in soc_constraint_ids:
            inverted_dual_vars[constr_id] = []

        for var_id in solution.dual_vars:
            if var_id in psd_constraint_ids:
                # Invert the PSD dual variables
                psd_dual_var = solution.dual_vars[var_id]

                # The first row of the PSD dual variable is the SOC dual variable.
                # This row corresponds to [A,B] part of the constraint matrix M which is simply the
                # concatenation of the scalar and vector terms of the original SOC constraint.
                soc_dual_var = psd_dual_var[0]

                # Append it to the list for the appropriate SOC constraint.
                soc_var_id = soc_id_from_psd[var_id]
                inverted_dual_vars[soc_var_id].append(soc_dual_var)
            else:
                # Include non-PSD dual variables without inversion.
                inverted_dual_vars[var_id] = solution.dual_vars[var_id]

        # Since SOC constraints sometimes have multiple packed constraints,
        # we pack their corresponding dual variables by stacking them.
        for var_id in inverted_dual_vars:
            if var_id in soc_constraint_ids:
                # The PSD representation requires scaling the dual variables by 2
                # to get the SOC dual variable.
                inverted_dual_vars[var_id] = 2 * np.hstack(inverted_dual_vars[var_id])

        solution.dual_vars = inverted_dual_vars
        return solution
