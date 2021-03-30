"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from cvxpy import *
import cvxopt
# Problem data
T = 10
n,p = (10,5)
A = cvxopt.normal(n,n)
B = cvxopt.normal(n,p)
x_init = cvxopt.normal(n)
x_final = cvxopt.normal(n)

# Object oriented optimal control problem.
class Stage:
    def __init__(self, A, B, x_prev):
        self.x = Variable(n)
        self.u = Variable(p)
        self.cost = sum(square(self.u)) + sum(abs(self.x))
        self.constraint = (self.x == A*x_prev + B*self.u)

stages = [Stage(A, B, x_init)]
for i in range(T):
    stages.append(Stage(A, B, stages[-1].x))

obj = sum(s.cost for s in stages)
constraints = [stages[-1].x == x_final]
map(constraints.append, (s.constraint for s in stages))
print(Problem(Minimize(obj), constraints).solve())
