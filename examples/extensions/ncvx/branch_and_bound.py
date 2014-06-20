import cvxopt
import cvxpy.problems.problem as problem
import cvxpy.settings as s
from boolean import Boolean

def branch(booleans):
    bool_vals = (b for b in booleans if not b.fix_values)
    # pick *a* boolean variable to branch on
    # choose the most ambivalent one (smallest distance to 0.5)
    # NOTE: if there are no boolean variables, will never branch
    return min(bool_vals, key=lambda x: abs(x.value - 0.5))

def bound(prob, booleans):
    # relax boolean constraints
    for bool_var in booleans: bool_var.relax()
    # solves relaxation
    lower_bound = prob._solve()
    if isinstance(lower_bound, str):
        lower_bound = float('inf')

    # round boolean variables and re-solve to obtain upper bound
    for bool_var in booleans: bool_var.round()
    upper_bound = prob._solve()
    if isinstance(upper_bound, str):
        upper_bound = float('inf')

    return {'gap': upper_bound - lower_bound,
            'ub': upper_bound,
            'lb': lower_bound,
            'obj': upper_bound,
            'sol': map(lambda x: x.value, booleans)}

def solve_wrapper(prob, i, booleans, depth, epsilon):
    if i > depth: return None

    # branch
    branch_var = branch(booleans)

    # try true branch
    branch_var.set(True)
    true_branch = bound(prob, booleans)

    # try false branch
    branch_var.set(False)
    false_branch = bound(prob, booleans)

    # keep track of best objective so far
    if true_branch['obj'] < false_branch['obj']:
        solution = true_branch
    else:
        solution = false_branch

    # update the bound
    solution['lb'] = min(true_branch['lb'],false_branch['lb'])
    solution['ub'] = min(true_branch['ub'],false_branch['ub'])

    # check if gap is small enough
    solution['gap'] = solution['ub'] - solution['lb']
    if solution['gap'] < epsilon:
        branch_var.unset()
        return solution

    # if the gap isn't small enough, we will choose a branch to go down
    def take_branch(true_or_false):
        branch_var.set(true_or_false)
        if true_or_false is True: branch_bools = true_branch['sol']
        else: branch_bools = false_branch['sol']
        # restore the values into the set of booleans
        for b, value in zip(booleans,branch_bools):
            b.save_value(value)
        return solve_wrapper(prob, i+1, booleans, depth, epsilon)

    # partition based on lower bounds
    if true_branch['lb'] < false_branch['lb']:
        true_subtree = take_branch(True)
        false_subtree = take_branch(False)
    else:
        false_subtree = take_branch(False)
        true_subtree = take_branch(True)

    # propagate best solution up the tree
    if true_subtree and false_subtree:
        if true_subtree['obj'] < false_subtree['obj']:
            return true_subtree
        return false_subtree
    if not false_subtree and true_subtree: return true_subtree
    if not true_subtree and false_subtree: return false_subtree

    # return best guess so far
    return solution

def branch_and_bound(self, depth=5, epsilon=1e-3):
    objective, constr_map = self.canonicalize()
    dims = self._format_for_solver(constr_map, s.ECOS)

    variables = self.objective.variables()
    for constr in self.constraints:
        variables += constr.variables()

    booleans = [v for v in variables if isinstance(v, Boolean)]

    self.constraints.extend(b._LB <= b for b in booleans)
    self.constraints.extend(b <= b._UB for b in booleans)

    result = bound(self, booleans)

    # check if gap is small enough
    if result['gap'] < epsilon:
        return result['obj']
    result = solve_wrapper(self, 0, booleans, depth, epsilon)

    # set the boolean values to the solution
    for b, value in zip(booleans, result['sol']):
        b.save_value(value)
        b.fix_values = cvxopt.matrix(True, b.size)

    return result['obj']

# add branch and bound a solution method
problem.Problem.register_solve("branch and bound", branch_and_bound)
