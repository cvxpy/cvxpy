__author__ = 'Xinyue'

from cvxpy import *
from find_set import find_minset
from fix import fix_prob
import numpy as np
from initial import rand_initial
import cvxpy as cvx

def is_dmcp(prob):
    """
    :param prob: a problem
    :return: a boolean indicating if the problem is DMCP
    """
    min_sets = find_minset(prob)
    if len(min_sets[0]) == len(prob.variables()): # if the minimal set contains all vars
        return False
    else:
        return True

def bcd(prob, max_iter = 100, solver = 'SCS', mu = 5e-3, rho = 1.5, mu_max = 1e5, ep = 1e-3, lambd = 10, linearize = False, proximal = True):
    """
    call the solving method
    :param prob: a problem
    :param max_iter: maximal number of iterations
    :param solver: DCP solver
    :param mu: initial value of parameter mu
    :param rho: increasing factor for mu
    :param mu_max: maximal value of mu
    :param ep: precision in convergence criterion
    :param lambd: parameter lambda
    :param linearize: if prox-lin operator is used
    :param proximal: if proximal operator is used
    :return: it: number of iterations; max_slack: maximum slack variable
    """
    # check if the problem is DMCP, and find minimal sets to fix
    fix_sets = find_minset(prob)
    if len(fix_sets[0]) == len(prob.variables()):
        print "problem is not DMCP"
        return None
    #
    #result = None
    #if prob.objective.NAME == 'minimize':
    #    cost_value = float("inf") # record on the best cost value
    #else:
    #    cost_value = -float("inf")
    #var_solution = []
    #dccp_ini(prob, random = True)
    flag_ini = 0
    for var in prob.variables():
        if var.value is None:
            flag_ini = 1
            break
    if flag_ini:
        rand_initial(prob)

    result = _bcd(prob, fix_sets, max_iter, solver, mu, rho, mu_max, ep, lambd, linearize, proximal)
    #if result_temp is None:
    #    return None
    #if (prob.objective.NAME == 'minimize' and prob.objective.value<cost_value) \
    #    or (prob.objective.NAME == 'maximize' and prob.objective.value>cost_value): # find a better cost value
    #        result = result_temp # update the result
    #        cost_value = prob.objective.value # update the record on the best cost value
    #        var_solution = [] # store the new solution
    #        for var in prob.variables():
    #            var_solution.append(var.value)
    #if var_solution is not None:
    #    for idx, var in enumerate(prob.variables()):
    #        var.value = var_solution[idx]
    print "======= result ======="
    print "minimal sets:", fix_sets
    if flag_ini:
        print "initial point not set by the user"
    print "number of iterations:", result[0]+1
    print "maximum value of slack variables:", result[1]
    print "objective value:", prob.objective.value
    return result

def _bcd(prob, fix_sets, max_iter, solver, mu, rho, mu_max, ep, lambd, linear, proximal):
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
        for set in fix_sets:
            fix_set = [var for var in prob.variables() if var.id in set]
            # fix variables in fix_set
            fixed_p = fix_prob(prob,fix_set)
            # linearize
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
                print "max abs slack =", max_slack, "mu =", mu, "original objective value =", prob.objective.args[0].value, "fixed objective value =",fixed_p.objective.args[0].value, "status=", fixed_p.status
            else:
                print "original objective value =", prob.objective.args[0].value, "status=", fixed_p.status
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
    Add a slack variable to each constraint.
    For leq constraint, the slack variable is non-negative, and is on the right-hand side
    :param prob: a problem
    :param mu: weight of slack variables
    :return: a new problem with slack vars added, and the list of slack vars
    """
    var_slack = []
    new_constr = []
    for constr in prob.constraints:
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