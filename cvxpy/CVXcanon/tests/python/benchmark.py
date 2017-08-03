
# coding: utf-8

# These are a list of problems where parsing is slow in CVXPY.
# 
# The first four are all from the Convex.jl paper.
# 
# Be careful when profiling CVXPY that you create a new problem each time.
# When you solve a problem it caches the cone program matrices, so if you solve it again it doesn't run the matrix stuffing algorithm.

# In[9]:

# Preparation:
from cvxpy import *
import numpy


# In[10]:

# Summation.
n = 10000
x = Variable()
e = 0
for i in range(n):
    e = e + x
p = Problem(Minimize(norm(e-1,2)), [x>=0])


# In[11]:

# baseline cvxpy
settings.USE_CVXCANON = False


# In[12]:

get_ipython().run_cell_magic(u'timeit', u'', u"# This generates the cone program matrices but doesn't solve the cone program.\n# 99% of the time here is spent in the matrix stuffing algorithm (going from LinOps to the final matrix).\np = Problem(Minimize(norm(e-1,2)), [x>=0])\np.get_problem_data(ECOS)")


# In[13]:

# CVXPY backed with CVXCanon
settings.USE_CVXCANON = True


# In[15]:

get_ipython().run_cell_magic(u'timeit', u'', u'p = Problem(Minimize(norm(e-1,2)), [x>=0])\np.get_problem_data(ECOS)')


# You can replace %%timeit with %%prun to profile the script.

# In[16]:

# Indexing.
n = 10000
x = Variable(n)
e = 0
for i in range(n):
    e += x[i];


# In[17]:

# baseline cvxpy
settings.USE_CVXCANON = False


# In[18]:

get_ipython().run_cell_magic(u'timeit', u'', u'p = Problem(Minimize(norm(e-1,2)), [x>=0])\np.get_problem_data(ECOS)')


# In[19]:

# CVXPY backed with CVXCanon
settings.USE_CVXCANON = True


# In[34]:

get_ipython().run_cell_magic(u'timeit', u'', u'p = Problem(Minimize(norm(e-1,2)), [x>=0])\np.get_problem_data(ECOS)')


# In[35]:

# Transpose
n = 500
A = numpy.random.randn(n,n)


# In[36]:

# baseline cvxpy
settings.USE_CVXCANON = False


# In[37]:

get_ipython().run_cell_magic(u'timeit', u'', u"X = Variable(n,n)\np = Problem(Minimize(norm(X.T-A,'fro')), [X[1,1] == 1])\np.get_problem_data(ECOS)")


# In[38]:

# CVXPY backed with CVXCanon
settings.USE_CVXCANON = True


# In[39]:

get_ipython().run_cell_magic(u'timeit', u'', u"X = Variable(n,n)\np = Problem(Minimize(norm(X.T-A,'fro')), [X[1,1] == 1])\np.get_problem_data(ECOS)")


# In[40]:

# Matrix constraint.
# CVXPY actually does a pretty good job with this one.
# Convex.jl and CVX are slower (at least when they were profiled for the paper).
n = 500
A = numpy.random.randn(n,n)
B = numpy.random.randn(n,n)


# In[41]:

# baseline cvxpy
settings.USE_CVXCANON = False


# In[42]:

get_ipython().run_cell_magic(u'timeit', u'', u"X = Variable(n,n)\np = Problem(Minimize(norm(X-A,'fro')), [X == B])\np.get_problem_data(ECOS)")


# In[43]:

# CVXPY backed with CVXCanon
settings.USE_CVXCANON = True


# In[44]:

get_ipython().run_cell_magic(u'timeit', u'', u"X = Variable(n,n)\np = Problem(Minimize(norm(X-A,'fro')), [X == B])\np.get_problem_data(ECOS)")


# In[45]:

# Matrix product.
# This one is a bit different, because the issue is that the coefficient for A.T*X*A has n^4 nonzeros.
# A fix is to introduce the variable A.T*X = Y, and rewrite A.T*X*A as Y*A. 
# This will only add 2n^3 nonzeros.
n = 50
A = numpy.random.randn(n,n)


# In[46]:

# baseline cvxpy
settings.USE_CVXCANON = False


# In[47]:

get_ipython().run_cell_magic(u'timeit', u'', u"X = Variable(n,n)\np = Problem(Minimize(norm(X,'fro')), [A.T*X*A >= 1])\np.get_problem_data(ECOS)")


# In[48]:

# CVXPY backed with CVXCanon
settings.USE_CVXCANON = True


# In[49]:

get_ipython().run_cell_magic(u'timeit', u'', u"X = Variable(n,n)\np = Problem(Minimize(norm(X,'fro')), [A.T*X*A >= 1])\np.get_problem_data(ECOS)")


# In[50]:

# SVM with indexing.
def gen_data(n):
    pos = numpy.random.multivariate_normal([1.0,2.0],numpy.eye(2),size=n)
    neg = numpy.random.multivariate_normal([-1.0,1.0],numpy.eye(2),size=n)
    return pos, neg

N = 2
C = 10
pos, neg = gen_data(500)

w = Variable(N)
b = Variable()
xi_pos = Variable(pos.shape[0])
xi_neg = Variable(neg.shape[0])
cost = sum_squares(w) + C*sum_entries(xi_pos) + C*sum_entries(xi_neg)
constrs = []
for j in range(pos.shape[0]):
    constrs += [w.T*pos[j,:] - b >= 1 - xi_pos[j]]
    
for j in range(neg.shape[0]):
    constrs += [-(w.T*neg[j,:] - b) >= 1 - xi_neg[j]]


# In[51]:

# baseline cvxpy
settings.USE_CVXCANON = False


# In[52]:

get_ipython().run_cell_magic(u'timeit', u'', u'p = Problem(Minimize(cost), constrs)\np.get_problem_data(ECOS)')


# In[53]:

# CVXPY backed with CVXCanon
settings.USE_CVXCANON = True


# In[54]:

get_ipython().run_cell_magic(u'timeit', u'', u'p = Problem(Minimize(cost), constrs)\np.get_problem_data(ECOS)')


# In[55]:




# In[ ]:



