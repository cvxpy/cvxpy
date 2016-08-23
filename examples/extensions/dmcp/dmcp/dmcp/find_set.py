__author__ = 'Xinyue'

from fix import fix
from fix import fix_prob
import numpy as np

def find_minimal_sets(prob):
    """
    find all minimal sets of a problem
    :param prob: a problem
    :return: result: a list of minimal sets,
    each is a set of indexes of variables in prob.variables()
    """
    if prob.is_dcp():
        return []
    maxsets = find_maxset_graph(prob)
    if maxsets is None:
        return [[i for i in range(len(prob.variables()))]]
    else:
        result = []
        for maxset in maxsets:
            maxset_id = [var.id for var in maxset]
            fix_id = [var.id for var in prob.variables() if var.id not in maxset_id]
            prob_var_id = [var.id for var in prob.variables()]
            fix_idx = [prob_var_id.index(varid) for varid in fix_id]
            result.append(fix_idx)
        return result


def find_maxset_graph(prob):
    """
    Analyze a problem by a graph to find maximum subsets of variables,
    so that the problem is dcp in each subset
    :param prob: Problem
    :return: a list of subsets of Variables, or None
    """
    if prob.is_dcp():
        return [prob.variables()]
    # graph of conflict vars
    node_num = len(prob.variables())
    t = np.zeros((node_num,node_num)) # table of edges
    varid = [var.id for var in prob.variables()]
    t = search_conflict(prob.objective.args[0],t,varid) # search conflicts in objective function
    for con in prob.constraints:
        t = search_conflict(con.args[0] + con.args[1],t,varid) # search conflicts in each constraint
    if not sum(np.diag(t)) == 0: # graph has self-loop <=> not dmcp
        return None
    return find_all_MIS(prob.variables(), t, prob)

def find_all_MIS(V,g,prob):
    """
    find {V1, V2, ..., Vk} such that:
        V1, ..., Vk are k maximal independent subsets of V on graph g that have the k-largest cardinalities;
        the union of V1, ..., Vk covers V;
        the union of V1, ..., V(k-1) cannot cover V
    :param g: graph
    :param V: set of all vertices
    :return: a list of maximal independent sets
    """
    i_subsets = find_all_iset(V,g)
    subsets_len = [len(subset) for subset in i_subsets]
    sort_idx = np.argsort(subsets_len) # sort the subsets by card
    result = []
    #U = [] # union of all collected vars
    for count in range(1,len(sort_idx)+1): # collecting from the subsets with largest card
        flag = 1
        for subs in result:
            if is_subset(i_subsets[sort_idx[-count]], subs): # the current one is a subset of a previously collected one
                flag = 0
                break
        if flag:
            set_id = [var.id for var in i_subsets[sort_idx[-count]]]
            fix_set = [var for var in prob.variables() if var.id not in set_id]
            if fix_prob(prob, fix_set).is_dcp():
                result.append(i_subsets[sort_idx[-count]])
                #U = union(U, i_subsets[sort_idx[-count]])
            else:
                return None
    #if is_subset(V,U): # the collected vars cover all vars
    return result
    #else:
    #    return None


def find_all_iset(V,g):
    """
    find all independent subsets
    :param V: vertex set
    :param g: graph
    :return: a list of independent subsets
    """
    subsets = find_all_subsets(V)
    result = []
    V_id = [var.id for var in V]
    for subset in subsets:
        subset_id = [var.id for var in subset]
        subset_ind = [V_id.index(i) for i in subset_id]
        if is_independent(subset_ind, g):
            result.append(subset)
    return result

def is_independent(s,g):
    """
    if a subset of vertices is independent on a graph
    :param s: a subset of vertices represented by indices
    :param g: graph
    :return: boolean
    """
    if sum([g[i,j] for i in s for j in s]) == 0:
        return True
    else:
        return False

def find_all_subsets(s):
    """
    find all subsets of a set, except for the empty set
    :param s: a set represented by a list
    :return: a list of subsets
    """
    subsets = []
    N = np.power(2,len(s))
    for n in range(N-1): # each number represent a subset
        set = [] # the subset corresponding to n
        binary_ind = np.binary_repr(n+1) # binary
        for idx in range(1,len(binary_ind)+1): # each bit of the binary number
            if binary_ind[-idx] == '1': # '1' means to add the element corresponding to that bit
                set.append(s[-idx])
        subsets.append(set)
    return subsets

def search_conflict(expr,t,varid):
    """
    search conflict variables in an expression
    :param expr: expression
    :param t: a table recording the conflict pairs
    :param varid: id of all vars in table t
    :return: table t
    """
    for arg in expr.args:
        t = search_conflict(arg,t,varid)
    #try:
    #    op = expr.OP_NAME
    #except AttributeError:
    #    op = None
    #if op == '*' and not expr.args[0].is_constant() and not expr.args[1].is_constant(): # multiplication of two vars
    if expr.is_atom_multiconvex() and not expr.args[0].is_constant() and not expr.args[1].is_constant():
        id1 = [var.id for var in expr.args[0].variables()] # var ids in left child node
        id2 = [var.id for var in expr.args[1].variables()]
        index1 = [varid.index(vi) for vi in id1] # table index in left child node
        index2 = [varid.index(vi) for vi in id2]
        for i in index1:
            for j in index2:
                t[i,j] = 1
                t[j,i] = 1
    return t

def is_intersect(set1, set2):
    """
    if the intersection of set1 and set2 is empty
    :param set1: a list of vars
    :param set2: a list of vars
    :return: boolean
    """
    id1 = [var.id for var in set1]
    id2 = [var.id for var in set2]
    flag = 0
    for id_1 in id1:
        for id_2 in id2:
            if id_1 == id_2:
                flag = 1
                return flag
    return flag

def union(set1, set2):
    """
    the union of set1 and set2
    :param set1: a list of vars
    :param set2: a list of vars
    :return: a list of vars
    """
    result = set1
    id1 = [var.id for var in set1]
    for var in set2:
        if not var.id in id1:
            result.append(var)
    return result


def find_maxset_prob(prob,vars,current=[]):
    """
    Analyze a problem to find maximum subsets of variables,
    so that the problem is dcp restricting on each subset
    :param prob: Problem
    :return: a list of subsets of Variables, or None
    """
    if prob.is_dcp():
        return [prob.variables()]
    result = []
    next_level = []
    for var in vars:
        vars_active = erase(vars,var) # active variables
        if vars_active == []:  # an empty list indicates that the problem is not multi-convex
            return None
        # if the set of active vars is not a subset of the current result
        if all([not is_subset(vars_active, current_set) for current_set in current]):
            vars_active_id = [var.id for var in vars_active]
            fix_vars_temp = [var for var in prob.variables() if not var.id in vars_active_id]
            if fix_prob(prob,fix_vars_temp).is_dcp() == True:
                result.append(vars_active) # find a subset
                current.append(vars_active)
            else:
                next_level.append(vars_active) # to be decomposed in the next level
    for set in next_level:
        result_temp = find_maxset_prob(prob,set,current)
        if result_temp is None:
            return None
        else:
            for set in result_temp:
                result.append(set)
    return result

def find_dcp_maxset(expr,vars,current=[]):
    """
    find maximum subsets of variables, so that expr is a dcp expression within each subset
    :param expr: an expression
    :param vars: variables that are not fixed
    :param current: current list of subsets
    :return: a list of subsets of variables and each subset is a list, or None
    """
    if expr.is_dcp():
        return [expr.variables()]
    result = []
    next_level = []
    for var in vars:
        vars_active = erase(vars,var) # active variables
        if vars_active == []:  # an empty list indicates that the expression is not multi-dcp
            return None
        # if the set of active vars is not a subset of the current result
        if all([not is_subset(vars_active, current_set) for current_set in current]):
            vars_active_id = [var.id for var in vars_active]
            fix_vars_temp = [var for var in expr.variables() if not var.id in vars_active_id]
            if fix(expr,fix_vars_temp).is_dcp() == True:
                result.append(vars_active) # find a subset
                current.append(vars_active)
            else:
                next_level.append(vars_active) # to be decomposed in the next level
    for set in next_level:
        result_temp = find_dcp_maxset(expr,set,current)
        if result_temp is None:
            return None
        else:
            for set in result_temp:
                result.append(set)
    return result

def find_dcp_set(expr, vars):
    """
    find subsets of variables, so that expr is a dcp expression within each subset
    :param expr:
    :param vars: variables that are not fixed
    :return: a list of subsets of variables and each subset is a list, or None
    """
    if vars == []:  # an empty list indicates that the expression is not multi-dcp
        return None
    vars_id = [var.id for var in vars]
    fix_vars = [var for var in expr.variables() if not var.id in vars_id]
    if fix(expr,fix_vars).is_dcp() == True:
        return [vars]
    else:
        result = []
        for var in vars: # erase each variable from vars
            vars_temp = erase(vars,var) # active variables
            result_temp = find_dcp_set(expr,vars_temp)
            if result_temp is None:
                return None
            for var_set in result_temp:
                result.append(var_set)
        return result

def is_subset(var_set1, var_set2):
    """
    :param var_set1: a list of variables
    :param var_set2: a list of variables
    :return: a boolean indicating if var_set1 is a subset of var_set2
    """
    if var_set2 == []:
        return False
    if var_set1 == []:
        return True
    var_set1_id = [var.id for var in var_set1]
    var_set2_id = [var.id for var in var_set2]
    flag = True
    for var_1_id in var_set1_id:
        if not var_1_id in var_set2_id:
            flag = False
            return flag
    return flag

def erase(vars,var):
    """
    erase var from a set of variables vars
    :param vars: a non-empty set of variables
    :param var: the variable to be erased from the set
    :return: a set of variables
    """
    return [v for v in vars if v != var]

'''
def find_maxset2(prob):
    if prob.is_dcp():
        return [prob.variables()]
    expr = prob.objective.args[0]
    for con in prob.constraints:
        expr += con.args[0]
        expr += con.args[1]
    return find_maxset_expr2(expr)
'''
'''
def find_maxset_expr2(expr):
    if isinstance(expr, Variable):
        return [[expr]]
    try:
        op = expr.OP_NAME
    except AttributeError:
        op = None
    if op == '*' and not expr.args[0].is_constant() and not expr.args[1].is_constant(): # a bi-convex node
        list1 = find_maxset_expr2(expr.args[0])
        list2 = find_maxset_expr2(expr.args[1])
        if list1 is None or list2 is None:
            return None
        for set1 in list1:
            for set2 in list2:
                if is_intersect(set1, set2): # if any two sets intersect
                    return None
        return list1+list2
    else: # a convex node
        list1 = find_maxset_expr2(expr.args[0])
        list2 = find_maxset_expr2(expr.args[1])
        if list1 is None or list2 is None:
            return None
        result = []
        t = np.zeros((len(list1), len(list2))) # table of indicator of intersection
        for set1 in list1:
            for set2 in list2:
                if is_intersect(set1,set2):
                    t[list1.index(set1),list2.index(set2)] = 1
        #for set1 in list1:
        #    for set2 in list2:

        return result
'''