__author__ = 'Xinyue'
from cvxpy import *
import numpy as np

def rand_initial(prob):
    """
    set random values to all variables
    :param prob: a problem
    :return:
    """
    for var in prob.variables():
        if var.sign == "POSITIVE":
            var.value = np.random.rand(var._rows,var._cols)
        else:
            var.value = np.random.randn(var._rows,var._cols)

def rand_initial_proj(self, times = 1, random = 1):
    """
    random initial value with projection
    :param times: number of random projections for each variable
    :param random: mandatory random initial values
    """
    dom_constr = self.objective.args[0].domain # domain of the objective function
    for arg in self.constraints:
        for l in range(2):
            for dom in arg.args[l].domain:
                dom_constr.append(dom) # domain on each side of constraints
    var_store = [] # store initial values for each variable
    init_flag = [] # indicate if any variable is initialized by the user
    for var in self.variables():
        var_store.append(np.zeros((var._rows,var._cols))) # to be averaged
        init_flag.append(var.value is None)
    # setup the problem
    ini_cost = 0
    var_ind = 0
    value_para = []
    for var in self.variables():
        if init_flag[var_ind] or random: # if the variable is not initialized by the user, or random initialization is mandatory
            value_para.append(Parameter(var._rows,var._cols))
            ini_cost += pnorm(var-value_para[-1],2)
        var_ind += 1
    ini_obj = Minimize(ini_cost)
    ini_prob = Problem(ini_obj,dom_constr)
    # solve it several times with random points
    for t in range(times): # for each time of random projection
        count_para = 0
        var_ind = 0
        for var in self.variables():
            if init_flag[var_ind] or random: # if the variable is not initialized by the user, or random initialization is mandatory
                value_para[count_para].value = np.random.randn(var._rows,var._cols)*10 # set a random point
                count_para += 1
            var_ind += 1
        ini_prob.solve()
        var_ind = 0
        for var in self.variables():
            var_store[var_ind] = var_store[var_ind] + var.value/float(times) # average
            var_ind += 1
    # set initial values
    var_ind = 0
    for var in self.variables():
        if init_flag[var_ind] or random:
            if var.sign == 'POSITIVE':
                var.value = np.abs(var_store[var_ind])
            else:
                var.value = var_store[var_ind]
            var_ind += 1