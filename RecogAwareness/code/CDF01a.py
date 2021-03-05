"""
Explore general discrete probability based on cdf

    Pr[x1 <= X1 <= x1+1] = cdf(x1+1) - cdf(x1)
    Pr[x1 <= X1 <= x1+1, 
       x2 <= X2 <= x2+1] = cdf(x1+1,x2+1) - cdf(x1+1,x2) - cdf(x1,x2+1) + cdf(x1,x2)
    ...
    Pr[x1 <= X1 <= x1+1, ... xd <= Xd <= xd+1] = <general formulation>

    Notation:
        P1 := Pr[x1 <= X1 <= x1+1]
        P2 := Pr[x1 <= X1 <= x1+1, x2 <= X2 <= x2+1]
        ...
        Pd := Pr[x1 <= X1 <= x1+1, ..., xd <= Xd <= xd+1]
        [1] := cdf(x1+1)
        [0] := cdf(x1)
        [11] := cdf(x1+1,x2+1)
        ...
        [a1 a2 ... ad] := cdf(x1+a1,x2+a2,...,xd+ad)

    View 1:
        P1 = [1] - [0]
        P2 = 1:P1 - 0:P1 = ([1:1] - [1:0]) - ([0:1] - [0:0]) = [11]-[10]-[01]+[00]
        P3 = 1:P2 - 0:P2 = ... = [111]-[110]-[101]+[100] - ([011]-[010]-[001]+[000])
        ...
        Pd = 1:P(d-1) - 0:P(d-1)
    
    View 2:
        P1 = s[1] s[0]
        P2 = s[11] s[10] s[01] s[00]
        P3 = s[111] s[110] s[101] s[100] s[011] s[010] s[001] s[000]
        ...
        Pd = s[binary 2^d - 1] s[binary 2^d - 2] ... s[binary 2^d - 2^d = binary 0]
        where s = + when sum([binary]) % 2 == d % 2
"""

import numpy as np
import scipy
from scipy.stats import multivariate_normal

'''
Simulator class
'''

class dnorm:
    def __init__(self, mean=[3,4], cov=[[1,0],[0,1]]):
        self.mean=mean
        self.cov=cov
        self.N = None
        self.D = None
        self.data = None
        
        
    def gen_data(self, num):
        cv = np.random.multivariate_normal(self.mean, self.cov, num)
        #print('cv = ', cv)
        dv = np.floor(cv)
        #print('dv = ', dv)
        self.data=dv
        self.N, self.D = self.data.shape
        
        
    def countp(self, x):
        return sum(np.sum(self.data == x, axis=1) == self.D)/self.N

'''
Test facilities
'''

def _inc(tpoints, tspace):
    D = len(tpoints)
    
    d = 0
    while d < D:
        if tpoints[d] + 1 < len(tspace[d]):
            tpoints[d] += 1
            break
        tpoints[d] = 0
        d += 1
        
    return tpoints

def prepare_testgrid(mins=[-1, -2], maxs=[1,3], res=[4,5]):
    D = len(mins)

    # Find ticks of every dimension
    dticks = []
    tids = []
    N = 1
    for d in range(D):
        ts = np.linspace(mins[d], maxs[d], res[d])
        dticks.append(ts)
        
        N *= len(ts)
        tids.append(0)
        
    # Generate testgrid x
    x_grid = np.zeros((N, D))
        
    for n in range(N):
        for d in range(D):
            tid = tids[d]
            x_grid[n,d] = dticks[d][tid]
        
        tids = _inc(tids, dticks)
            
    return x_grid


    '''
    Discrete Prob function under test
    '''

def dprob2D(x, cum_distf):

    p22 = cum_distf(x + [1, 1])
    p21 = cum_distf(x + [1, 0])
    p12 = cum_distf(x + [0, 1])
    p11 = cum_distf(x + [0, 0])

    return p22 - p21 - p12 + p11

'''
Compact test round 
'''
def test(mean, cov, num, dprob_calcf, mins=None, maxs=None, res=None):

    D = len(mean)

    if mins == None:
        mins = mean - 3* np.sqrt(np.diag(cov))
    
    if maxs == None:
        maxs = mean + 3* np.sqrt(np.diag(cov))

    if res == None:
        num_cases = 100 # default number of test cases
        r = int(np.ceil(np.exp(np.log(num_cases)/D)))
        res = r * np.ones(D)

    x_grid = prepare_testgrid(mins, maxs, res)

    sim = dnorm(mean, cov)
    sim.gen_data(num)

    cdf = lambda x: multivariate_normal.cdf(x, mean, cov)

    SEs = []
    for x in x_grid:
        p1 = sim.countp(x)
        p2 = dprob_calcf(x, cdf)
        se = (p1 - p2)**2
        SEs.append(se)
    
    return SEs


if __name__ == '__main__':
    mu = [5, 5]
    Sigma = [[9, 0],[0, 9]]
    
    sim = dnorm(mu, Sigma)
    sim.gen_data(1000)

    cdf = lambda x: multivariate_normal.cdf(x, mean=mu, cov=Sigma)

    x = np.array([4,8])
    p1 = sim.countp(x)
    p2 = dprob2D(x, cdf)
    se = (p1 - p2)**2
    print('count = ', p1, '; calc = ', p2, '; diff = ', se )

    se = test(mu, Sigma, 2000, dprob2D)

    print('Test MSE = ')
    print(np.mean(se))