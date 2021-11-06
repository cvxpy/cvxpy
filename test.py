import time                                                                      
                                                                                 
import cvxpy as cp                                                               
import matplotlib.pyplot as plt                                                  
import numpy as np                                                               
import scipy.sparse as sparse                                                    
from tqdm.notebook import tqdm                                                   
                                                                                 
                                                                                 
def construct_lp(n):                                                             
    c = sparse.csr_matrix((n, 1))  # timings are similar if c is a dense numpy array
    x = cp.Variable(n)                                                           
    return cp.Problem(cp.Minimize(c.T @ x), [x >= 0])                            
                                                                                 
n_range = list(map(int, np.logspace(1, 8, 8)))                                   
solvers = {"OSQP": [], "SCS": []}                                                
for solver in solvers:                                                           
    compile_times = solvers[solver]                                              
    for n in tqdm(n_range):                                                      
        problem = construct_lp(n)                                                
        start = time.time()                                                      
        problem.get_problem_data(solver, verbose=True)                           
        end = time.time()                                                        
        print(end - start)                                                       
        compile_times.append(end - start)                                        
                                                                                 
for solver in solvers:                                                           
    compile_times = solvers[solver]                                              
    plt.plot(n_range, compile_times, label=solver, marker='o')                   
    plt.xscale('log')                                                            
    plt.yscale('log')                                                            
plt.legend()                                                                     
plt.show()
