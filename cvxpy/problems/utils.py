import cvxopt

def mat_rows(A):
    """ Yields the individual rows of a matrix A.
    """
    for i in xrange(A.size[0]):
        yield A[i,:]

def normalize_data(A,b):
    """ This function normalizes the linear system
            A*x == b
        by setting the maximum absolute value of each row of A to 1. Also
        normalizes each entry of b by this amount.
    """
    elements = [(e/cvxopt.max(abs(e)), 
                 bi/(cvxopt.max(abs(e))), 
                 cvxopt.max(abs(e))) 
                for (e,bi) in zip(mat_rows(A),b) if cvxopt.max(abs(e)) != 0]
    reduced_rows, reduced_b, scale = zip(*elements) # unpack
        
    return reduced_rows, reduced_b, scale

def uniqify(a_list):
    """ This function uniqifies a list while preserving order.
    """
    # order preserving
    seen = set()
    for x in a_list:
        if x in seen:
            continue
        seen.add(x)
        yield x
        
def remove_redundant_rows(A,b):
    """ This function removes redundant constraints from the linear system
            A*x == b
        It can be thought of as a presolve.
    """
    if A.size[0] > 0:
        Aeq = [Equation(ai,bi,ci) for ai,bi,ci in zip(*normalize_data(A,b))]
        Aeq = list(uniqify(Aeq))
        rows = [elem.scale * elem.ai for elem in Aeq]
        consts = [elem.scale * elem.bi for elem in Aeq]
    
        return cvxopt.sparse(rows), cvxopt.matrix(consts)
    else:
        return A, b
    
class Equation(object):
    """ An Equation consists of a row and a constant: it represents
            a_i^T * x == b_i
        "Magic" happens in the hash function which works by using 
        (1:n)^T*(a_i) as the hash. Isn't terribly unique, but works for
        most things.
    """
    def __init__(self, ai, bi, ci):
        self.ai = cvxopt.sparse(ai)
        self.bi = bi
        self.scale = ci
        super(Equation,self).__init__()
    
    def __hash__(self):
        # uses [1,2,3,4,5,...] * x as hash function
        n = self.ai.size[1]
        prod = (self.ai.J + 1).T * self.ai.V
        # round to the 9th digit
        prod = abs(round(prod[0]*1e9)/1e9)
        return prod.__hash__()
        
    def __eq__(self, other):
        same = all(x <= 1e-9 for x in abs(self.ai - other.ai)) and abs(self.bi - other.bi) <= 1e-9
        diff_sign = all(x <= 1e-9 for x in abs(self.ai + other.ai)) and abs(self.bi + other.bi) <= 1e-9
        return same or diff_sign
        
    def __repr__(self):
        return str(self.ai)
        
    