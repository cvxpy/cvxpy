from cvxpy.problems.objective import Minimize, Maximize
from cvxpy.problems.problem import Problem
import cvxpy.utilities as utils
from cvxpy.atoms.atom import Atom
from cvxpy.expressions.variables import Variable
from cvxpy.expressions.constants import Constant
from cvxpy.lin_ops import lin_op, lin_utils
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
import cvxpy.expressions.types
from cvxpy.constraints import *

import copy, Queue

class PartialProblem(Atom):
    
    def __init__(self, prob, opt_vars, dont_opt_vars):

        self._my_prob = copy.deepcopy(prob)

        self._opt_vars = opt_vars
        self.create_new_opt_vars()
        
        self._dont_opt_vars = dont_opt_vars


        self.args = []
        
        self.validate_arguments()
        self.init_dcp_attr()
        self.subexpressions = self.args

    # Start overrides
    def __call__(self, *args):        
        self._dont_opt_vals = list(args)        
        return self
    
    def init_dcp_attr(self): # Called by ctor
        self._dcp_attr = utils.DCPAttr(self.sign_from_args(),
                                       self.func_curvature(),
                                       self.shape_from_args())
        
    def shape_from_args(self):
        return utils.Shape(1,1)

    def sign_from_args(self):
        return self._my_prob.objective._expr._dcp_attr.sign

    def func_curvature(self):
        if isinstance(self._my_prob.objective, Minimize):
            return utils.curvature.Curvature.CONVEX
        elif isinstance(self._my_prob.objective, Maximize):
            return utils.curvature.Curvature.CONCAVE            
        else:            
            raise Exception("You called partial_optimize with a Problem object that contains neither a Minimize nor a Maximize statement; this is not supported.")

    def monotonicity(self):
        return utils.monotonicity.NONMONOTONIC

    def get_data(self): # Called by atoms/atom.py:canonicalize()
        data = {}
        data["dont_opt_vals"] = self._dont_opt_vals
        data["dont_opt_vars"] = self._dont_opt_vars
        data["opt_vars"] = self._opt_vars        
        data["new_opt_vars"] = self._new_opt_vars
        data["objf"] = self._my_prob.objective
        data["constrs"] = self._my_prob.constraints
        return data

    def variables(self): # Works similar to atoms/atom.py:variables; called by problems/problem.py

        var_list = []

        for obj in self._dont_opt_vals + self._new_opt_vars:
            try:
                var_list += obj.variables()
            except Exception:
                pass
        
        return list(set(var_list))

    def name(self):
        return self.__str__()

    def __str__(self):
        return "PartialProblem(%s)" % self._my_prob.__str__()

    def __repr__(self):
        return self.__str__()
    
    def __deepcopy__(self, memo):
        pp = PartialProblem(self._my_prob, self._opt_vars, self._dont_opt_vars)
        pp._dont_opt_vals = self._dont_opt_vals
        return pp
    # End overrides
        
    def create_new_opt_vars(self):

        self._new_opt_vars = []

        for opt_var in self._opt_vars:
            dims = opt_var.size
            self._new_opt_vars += [opt_var.__class__(dims[0], dims[1])]

    @staticmethod
    def create_constr(constr, *args):

        if isinstance(constr, LeqConstraint):
            return [LeqConstraint(args[0], args[1])]

        elif isinstance(constr, EqConstraint):
            return [EqConstraint(args[0], args[1])]

        else:
            raise Exception("You created a Problem object that you passed into partial_optimize that contained a constraint that is unsupported right now.")

    @staticmethod
    def clamp_prob(dont_opt_vars, dont_opt_vals, opt_vars, new_opt_vars, objf, constrs):

        clamp_vars = dont_opt_vars + opt_vars
        clamps = dont_opt_vals + new_opt_vars


        objf_expr_clamped = PartialProblem.clamp_vars(objf._expr, clamp_vars, clamps)

        constrs_clamped = []
        for constr in constrs:
            constr_lh_exp_clamped = PartialProblem.clamp_vars(constr.lh_exp, clamp_vars, clamps)
            constr_rh_exp_clamped = PartialProblem.clamp_vars(constr.rh_exp, clamp_vars, clamps)
            constrs_clamped += PartialProblem.create_constr(constr, constr_lh_exp_clamped, constr_rh_exp_clamped)


        sense = objf.__class__
        prob_clamped = Problem(sense(objf_expr_clamped), constrs_clamped)
        return prob_clamped
    
    @staticmethod
    def graph_implementation(arg_objs, size, data=None): # arg_objs == self.args, but after calling canonicalize on each element == ignored here
        
        dont_opt_vals = data["dont_opt_vals"]
        dont_opt_vars = data["dont_opt_vars"]
        opt_vars = data["opt_vars"]
        new_opt_vars = data["new_opt_vars"]        
        objf = data["objf"]
        constrs = data["constrs"]

        prob_clamped = PartialProblem.clamp_prob(dont_opt_vars, dont_opt_vals, opt_vars, new_opt_vars, objf, constrs)
        return prob_clamped.canonicalize()

    def __eq__(self, pp): # pp == another PartialProblem
        return self._my_prob == pp._my_prob
        
    @Atom.numpy_numeric
    def numeric(self, values):

        dont_opt_vals = list(values)

        data = self.get_data()
        opt_vars = data["opt_vars"]        
        dont_opt_vars = data["dont_opt_vars"]        
        new_opt_vars = data["new_opt_vars"]        
        objf = data["objf"]
        constrs = data["constrs"]

        
        prob_clamped = PartialProblem.clamp_prob(dont_opt_vars, dont_opt_vals, opt_vars, new_opt_vars, objf, constrs)
        prob_clamped.solve()
        return prob_clamped.value

    class QueueItem:
        def __init__(self, item, parent=None, idx=None):
            self._item = item
            self._parent = parent
            self._idx = idx

    @staticmethod
    def get_idx_of_var_in_var_list(match_var, var_list):

        match_var_guid = ""
        if isinstance(match_var, Variable):
            match_var_guid = match_var.id
        else: # Old: == isinstance(match_var, lin_op.LinOp):
            match_var_guid = match_var.data
        
        
        for idx, var in enumerate(var_list):
            if match_var_guid == var.id:
                return idx
        return -1

    @staticmethod
    def is_var(expr):        
        if isinstance(expr, Variable) or \
            (isinstance(expr, lin_op.LinOp) and expr.type == lin_op.VARIABLE):     
            return True
        else:
            return False
    
    @staticmethod
    def get_clamp(clamps, idx):

        clamp = clamps[idx]

        if PartialProblem.is_var(clamp):
            return clamp
        else:
            return cvxpy.expressions.types.constant()(clamp)

    @staticmethod
    def clamp_vars(expr, clamp_vars, clamps):

        my_queue = Queue.Queue()
        my_queue.put(PartialProblem.QueueItem(expr))

        while True:
            
            queue_item = my_queue.get()
            cur_expr = queue_item._item
            if PartialProblem.is_var(cur_expr):

                idx = PartialProblem.get_idx_of_var_in_var_list(cur_expr, clamp_vars)
                if idx is not -1:
                    
                    cur_expr_clamped = PartialProblem.get_clamp(clamps, idx)
                    
                    
                    parent = queue_item._parent
                    idx = queue_item._idx
                    
                    if parent is not None:
                        parent.args[idx] = cur_expr_clamped
                    else: # cur_expr is @ the root of the expr tree
                        expr = cur_expr_clamped

            else:
                if hasattr(cur_expr, "args") and len(cur_expr.args) > 0:
                    for idx, arg in enumerate(cur_expr.args):
                        my_queue.put(PartialProblem.QueueItem(arg, cur_expr, idx))


            if my_queue.empty():                
                break


        return expr

def partial_optimize(prob, opt_vars, dont_opt_vars):
    return PartialProblem(prob, opt_vars, dont_opt_vars)