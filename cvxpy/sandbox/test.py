from cyipopt import minimize_ipopt
from scipy.optimize import rosen, rosen_der

x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
res = minimize_ipopt(rosen, x0, jac=rosen_der)
print(res)