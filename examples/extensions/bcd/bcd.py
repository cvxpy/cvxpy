__author__ = 'Xinyue'

from cvxpy import *
from find_set import find_maxset_graph
from fix import fix_prob
import numpy as np
from initial import dccp_ini
import cvxpy as cvx

def bcd(prob, max_iter = 100, solver = 'SCS', mu=5e-3, rho=1.5, mu_max = 1e5, ep = 1e-3, random_ini = 1, random_times = 3, lambd = 10, linear=False, proximal=True):
    # check if the problem is DMCP, and find the maximal sets
    convex_sets = find_maxset_graph(prob)
    if convex_sets is None:
        print "problem is not DMCP"
        return None
    else:
        print "maximal sets:", [[var.name() for var in subset]for subset in convex_sets]
    #
    result = None
    if prob.objective.NAME == 'minimize':
        cost_value = float("inf") # record on the best cost value
    else:
        cost_value = -float("inf")
    if not random_ini:
        random_times = 1
    var_solution = []
    for t in range(random_times):
        if random_ini:
            dccp_ini(prob, random = random_ini)
        result_temp = _bcd(prob, convex_sets, max_iter, solver, mu, rho, mu_max, ep, lambd, linear, proximal)
        #result_temp = _joint(prob, convex_sets, max_iter, solver, mu, rho, mu_max, ep, lambd, linear, proximal)
        if result_temp is None:
            return None
        if (prob.objective.NAME == 'minimize' and prob.objective.value<cost_value) \
            or (prob.objective.NAME == 'maximize' and prob.objective.value>cost_value): # find a better cost value
                if t==0 or result[1]<1e-4: # first iteration; slack small enough
                    result = result_temp # update the result
                    cost_value = prob.objective.value # update the record on the best cost value
                    var_solution = [] # store the new solution
                    for var in prob.variables():
                        var_solution.append(var.value)
    if var_solution is not None:
        for idx, var in enumerate(prob.variables()):
            var.value = var_solution[idx]
    print "===="
    print "number of iterations:", result[0]
    print "max slack:", result[1]
    return result

def _joint(prob, convex_sets, max_iter, solver, mu, rho, mu_max, ep, lambd, linear, proximal):
    obj_pre = np.inf
    for it in range(max_iter):
        joint_obj = 0
        joint_constr = []
        var_slack = []
        for set in convex_sets:
            set_id = [var.id for var in set]
            fix_set = [var for var in prob.variables() if var.id not in set_id]
            # fix variables in fix_set
            fixed_p = fix_prob(prob,fix_set)
            # add slack variables
            fixed_p, var_sk = add_slack(fixed_p, mu)
            # adding to the joint problem
            joint_obj += fixed_p.objective.args[0]
            for con in fixed_p.constraints:
                joint_constr += [con]
            var_slack += var_sk
        for var in prob.variables():
            joint_obj += sum_entries(square(var-var.value))/float(2)/lambd
        joint_prob = Problem(Minimize(joint_obj), joint_constr)
        # solve
        joint_prob.solve(solver = solver)
        max_slack = 0
        if not var_slack == []:
            max_slack = np.max([np.max(abs(var).value) for var in var_slack])
        print "max abs slack =", max_slack, "mu =", mu, "obj_value =", joint_prob.value
        mu = min(mu*rho, mu_max) # adaptive mu
        if np.linalg.norm(obj_pre - prob.objective.args[0].value) <= ep and max_slack<=ep: # quit
            return it, max_slack
        else:
            obj_pre = prob.objective.args[0].value
    return it, max_slack


def _bcd(prob, convex_sets, max_iter, solver, mu, rho, mu_max, ep, lambd, linear, proximal):
    """
    block coordinate descent
    :param prob: Problem
    :param max_iter: maximum number of iterations
    :param solver: solver used to solved each fixed problem
    :return: it: number of iterations; max_slack: maximum slack variable
    """
    obj_pre = np.inf
    for it in range(max_iter):
        #print "======= iteration", it, "======="
        for set in convex_sets:
            set_id = [var.id for var in set]
            fix_set = [var for var in prob.variables() if var.id not in set_id]
            # fix variables in fix_set
            fixed_p = fix_prob(prob,fix_set)
            # linear
            if linear:
                fixed_p.objective.args[0] = linearize(fixed_p.objective.args[0])
            # add slack variables
            fixed_p, var_slack = add_slack(fixed_p, mu)
            # proximal operator
            if proximal:
                fixed_p = proximal_op(fixed_p, var_slack, lambd)
            # solve
            fixed_p.solve(solver = solver)
            max_slack = 0
            if not var_slack == []:
                max_slack = np.max([np.max(abs(var).value) for var in var_slack])
                print "max abs slack =", max_slack, "mu =", mu, "obj_value =", prob.objective.args[0].value, "fixed_p_value =",fixed_p.objective.args[0].value
            else:
                print "objective value =", prob.objective.args[0].value
        mu = min(mu*rho, mu_max) # adaptive mu
        if np.linalg.norm(obj_pre - prob.objective.args[0].value) <= ep and max_slack<=ep: # quit
            return it, max_slack
        else:
            obj_pre = prob.objective.args[0].value
    return it, max_slack

def linearize(expr):
    """Returns the tangent approximation to the expression.
    Gives an elementwise lower (upper) bound for convex (concave)
    expressions. No guarantees for non-DCP expressions.
    Args:
        expr: An expression.
    Returns:
        An affine expression.
    """
    if expr.is_affine():
        return expr
    else:
        tangent = expr.value
        if tangent is None:
            raise ValueError(
        "Cannot linearize non-affine expression with missing variable values."
            )
        grad_map = expr.grad
        for var in expr.variables():
            if var.is_matrix():
                flattened = np.transpose(grad_map[var])*vec(var - var.value)
                tangent = tangent + reshape(flattened, *expr.size)
            else:
                if var.size[1] == 1:
                    tangent = tangent + np.transpose(grad_map[var])*(var - var.value)
                else:
                    tangent = tangent + (var - var.value)*grad_map[var]
        return tangent

def add_slack(prob, mu):
    """
    Add a slack variable to each constraint that only has one or zero variable.
    For leq constraint, the slack variable is non-negative, and is on the right-hand side
    :param prob: Problem
    :return: Problem with slack variables, and a list of slack variables
    """
    var_slack = []
    new_constr = []
    for constr in prob.constraints:
        if len(constr.variables()) <= 1:
            row = max([constr.args[0].size[0], constr.args[1].size[0]])
            col = max([constr.args[0].size[1], constr.args[1].size[1]])
            if constr.OP_NAME == "<=":
                var_slack.append(NonNegative(row,col)) # NonNegative slack var
                left = constr.args[0]
                right =  constr.args[1] + var_slack[-1]
                new_constr.append(left<=right)
            elif constr.OP_NAME == ">>":
                var_slack.append(NonNegative(1)) # NonNegative slack var
                left = constr.args[0] + var_slack[-1]*np.eye(row)
                right =  constr.args[1]
                new_constr.append(left>>right)
            else: # equality constraint
                var_slack.append(Variable(row,col))
                left = constr.args[0]
                right =  constr.args[1] + var_slack[-1]
                new_constr.append(left==right)
        else:
            new_constr.append(constr)
    new_cost = prob.objective.args[0]
    if prob.objective.NAME == 'minimize':
        for var in var_slack:
            new_cost  =  new_cost + norm(var,1)*mu
        new_prob = Problem(Minimize(new_cost), new_constr)
    else: # maximize
        for var in var_slack:
            new_cost  =  new_cost - norm(var,1)*mu
        new_prob = Problem(Maximize(new_cost), new_constr)
    return new_prob, var_slack

def proximal_op(prob, var_slack, lambd):
    """
    proximal operator of the objective
    :param prob: problem
    :param var_slack: slack variables
    :param lambd: proximal operator parameter
    :return: a problem with proximal operator
    """
    new_cost = prob.objective.args[0]
    slack_id = [var.id for var in var_slack]
    for var in prob.variables():
        # add quadratic terms for all variables that are not slacks
        if not var.id in slack_id:
            new_cost = new_cost + square(norm(var - var.value,'fro'))/2/lambd
    prob.objective.args[0] = new_cost
    return prob

cvx.Problem.register_solve("bcd", bcd)