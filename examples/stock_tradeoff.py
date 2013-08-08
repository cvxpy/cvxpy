import cvxopt
import numpy
from cvxpy import *
from multiprocessing import Pool
from pylab import figure, show

class StockMarket(object):
    def __init__(self, num_assets, num_factors):
        self.num_assets = num_assets
        self.num_factors = num_factors

        self.expected_returns = cvxopt.exp( cvxopt.normal(self.num_assets) )
        self.factors = cvxopt.normal(num_assets, num_factors)
        self.asset_risk = cvxopt.uniform(num_assets)


class Portfolio(object):
    def __init__(self, budget, market):
        self.mu = market.expected_returns
        self.F = market.factors
        self.D = cvxopt.spdiag( market.asset_risk )
        self.x = Variable(market.num_assets)
        self.budget = budget
        self.gamma = Parameter()
        self.gamma.value = 1    # hack to ensure DCP rules hold

        # construct portfolio optimization problem
        self.p = Problem(
            Maximize(self.expected_return - self.gamma * self.variance),
            [sum(self.x) == self.budget, self.x >= 0]
          )

    @property
    def allocation(self):
        return self.x.value

    @property
    def expected_return(self):
        return self.mu.T * self.x

    @property
    def variance(self):
        return square(norm2(self.F.T * self.x)) + square(norm2(self.D * self.x))

    def allocate(self, tradeoff_parameter):
        self.gamma.value = tradeoff_parameter
        return self.p.solve()

# create a stock market and a portfolio
NYSE = StockMarket(num_assets = 5, num_factors=20)
my_portfolio = Portfolio(10, NYSE)

# encapsulate the portfolio run
def determine_allocation(x):
    my_portfolio.allocate(x)
    y = my_portfolio.allocation
    mu = my_portfolio.mu
    F = my_portfolio.F
    D = my_portfolio.D
    n = NYSE.num_assets
    expected_return, risk = mu.T*y, y.T*(F*F.T + D)*y
    return (expected_return[0], risk[0])

# create a pool of workers and a grid of gamma values
pool = Pool(processes = 4)
gammas = numpy.logspace(-1, 2, num=100)

mu, sigma = zip(*pool.map(determine_allocation, gammas))

# plot the result
fig = figure(1)
ax = fig.add_subplot(111)
ax.plot(mu, sigma)
ax.set_xlabel('expected return')
ax.set_ylabel('portfolio variance')

show()