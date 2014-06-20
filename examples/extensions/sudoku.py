from cvxpy import *
from ncvx.boolean import Boolean
import ncvx.branch_and_bound
import cvxopt
import cProfile, pstats
import numpy as np

n = 9
# 9x9 sudoku grid
numbers = [Boolean(n,n), Boolean(n,n), Boolean(n,n),
           Boolean(n,n), Boolean(n,n), Boolean(n,n),
           Boolean(n,n), Boolean(n,n), Boolean(n,n)]

# TODO: 9*[Boolean(9,9)] doesn't work....

solution = cvxopt.matrix([
    [0, 5, 2, 3, 7, 1, 8, 6, 4],
    [6, 3, 7, 8, 0, 4, 5, 2, 1],
    [1, 4, 8, 5, 2 ,6, 3, 0, 7],
    [4, 7, 1, 2, 3, 0, 6, 5, 8],
    [3, 6, 5, 1, 4, 8, 0, 7, 2],
    [8, 2, 0, 6, 5, 7, 4, 1, 3],
    [5, 1, 6, 7, 8, 3, 2, 4, 0],
    [7, 0, 3, 4, 6, 2, 1, 8, 5],
    [2, 8, 4, 0, 1, 5, 7, 3, 6]
])


# partial grid
known =[(0,6), (0,7), (1,4), (1,5), (1,8), (2,0), (2,2), (2,7), (2,8),
        (3,0), (3,1), (4,0), (4,2), (4,4), (4,6), (4,8), (5,7), (5,8),
        (6,0), (6,1), (6,6), (6,8), (7,0), (7,3), (7,4), (8,1), (8,2)]

def row(x,r):
    m, n = x.size
    for i in xrange(m):
        for j in xrange(n):
            if i == r: yield x[i,j]

def col(x,c):
    m, n = x.size
    for i in xrange(m):
        for j in xrange(n):
            if j == c: yield x[i,j]

def block(x,b):
    m, n = x.size
    for i in xrange(m):
        for j in xrange(n):
            # 0 block is r = 0,1, c = 0,1
            # 1 block is r = 0,1, c = 2,3
            # 2 block is r = 2,3, c = 0,1
            # 3 block is r = 2,3, c = 2,3
            if i // 3 == b // 3 and j // 3 == b % 3:
                yield x[i,j]

pr = cProfile.Profile()
pr.enable()
# create the suboku constraints
constraints = [sum(numbers) == 1]
for i in range(n):
    for num in range(n):
        constraints.append(sum(row(numbers[num], i)) == 1)
        constraints.append(sum(col(numbers[num], i)) == 1)
        constraints.append(sum(block(numbers[num], i)) == 1)
constraints.extend(numbers[solution[k]][k] == 1 for k in known)

# attempt to solve

p = Problem(Minimize(sum(map(square, [num[0,0] for num in numbers]))), constraints)
p.solve(method="branch and bound")
pr.disable()

ps = pstats.Stats(pr)
ps.sort_stats('tottime').print_stats(.5)

A = np.zeros((n, n))
for i, num in enumerate(numbers):
    A += i * num.value

print np.sum(A - solution)
