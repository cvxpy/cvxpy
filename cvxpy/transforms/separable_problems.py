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

from cvxpy.problems.problem import Problem
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.constants import Constant

import itertools
from scipy.sparse import dok_matrix, csgraph


def get_separable_problems(problem):
    """Return a list of separable problems whose sum is the original one.

    Parameters
    ----------
    problem : Problem
        A problem that consists of separable (sub)problems.

    Returns
    -------
    List
        A list of problems which are separable whose sum is the original one.
    """
    # obj_terms contains the terms in the objective functions. We have to
    # deal with the special case where the objective function is not a sum.
    if isinstance(problem.objective.args[0], cvxtypes.add_expr()):
        obj_terms = problem.objective.args[0].args
    else:
        obj_terms = [problem.objective.args[0]]
    # Remove constant terms, which will be appended to the first separable
    # problem.
    constant_terms = [term for term in obj_terms if term.is_constant()]
    obj_terms = [term for term in obj_terms if not term.is_constant()]

    constraints = problem.constraints
    num_obj_terms = len(obj_terms)
    num_terms = len(obj_terms) + len(constraints)

    # Objective terms and constraints are indexed from 0 to num_terms - 1.
    var_sets = [frozenset(func.variables()) for func in obj_terms + constraints
                ]
    all_vars = frozenset().union(*var_sets)

    adj_matrix = dok_matrix((num_terms, num_terms), dtype=bool)
    for var in all_vars:
        # Find all functions that contain this variable
        term_ids = [i for i, var_set in enumerate(var_sets) if var in var_set]
        # Add an edge between any two objetive terms/constraints sharing
        # this variable.
        if len(term_ids) > 1:
            for i, j in itertools.combinations(term_ids, 2):
                adj_matrix[i, j] = adj_matrix[j, i] = True
    num_components, labels = csgraph.connected_components(adj_matrix,
                                                          directed=False)

    # After splitting, construct subproblems from appropriate objective
    # terms and constraints.
    term_ids_per_subproblem = [[] for _ in range(num_components)]
    for i, label in enumerate(labels):
        term_ids_per_subproblem[label].append(i)
    problem_list = []
    for index in range(num_components):
        terms = [obj_terms[i] for i in term_ids_per_subproblem[index]
                 if i < num_obj_terms]
        # If we just call sum, we'll have an extra 0 in the objective.
        obj = sum(terms[1:], terms[0]) if terms else Constant(0)
        constrs = [constraints[i - num_obj_terms]
                   for i in term_ids_per_subproblem[index]
                   if i >= num_obj_terms]
        problem_list.append(Problem(problem.objective.copy([obj]), constrs))
    # Append constant terms to the first separable problem.
    if constant_terms:
        # Avoid adding an extra 0 in the objective
        sum_constant_terms = sum(constant_terms[1:], constant_terms[0])
        if problem_list:
            problem_list[0].objective.args[0] += sum_constant_terms
        else:
            problem_list.append(Problem(problem.objective.copy(
                [sum_constant_terms])))
    return problem_list
