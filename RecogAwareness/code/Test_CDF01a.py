from CDF01a import *


def dprobND(x, cumdistf):
    '''
    x input in np.array([x1, x2, ..., xd])
    cumdistf is cummulative distribution function, being able to evaluate cumdistf(x).
    * cumdistf(x) := cdf(x) := Pr[X <= x]

    This is a ground-up-attern algorithm.
    P1 = [1] - [0]
    ...
    Pd = 1:[P(d-1)] - 0:[P(d-1)]
    '''
    D = len(x)

    # Find pattern and signs
    Pat = [[1], [0]]
    Sign = [1, -1]
    for d in range(2, D + 1):

        NewPat = []
        NewSign = []

        for p in Pat:
            PA = [1]
            PA.extend(p)
            NewPat.append(PA.copy())

        for p in Pat:
            PB = [0]
            PB.extend(p)
            NewPat.append(PB.copy())

            Pat = NewPat

        for s in Sign:
            NewSign.append(s * 1)

        for s in Sign:
            NewSign.append(s * -1)

        Sign = NewSign

    # Calculate prob from cdf
    dprob = 0
    for i, p in enumerate(Pat):
        dprob += Sign[i] * cumdistf(x + np.array(p))

    return dprob


def dprob_bin(x, cumdistf):
    D = len(x)
    p = 0
    Bmax = 2 ** D
    for i in range(Bmax):
        # get the offset
        bcode = bin(Bmax - i - 1)[2:]
        offset = np.array([int(b) for b in bcode])

        # get the sign
        sign = 1
        if sum(offset) % 2 != D % 2:
            sign = -1

        p += sign * cumdistf(x + offset)

    return p

if __name__ == '__main__':
    mu = [5]
    Sigma = [3]  # [9, 0],[0, 9]]
    x = np.array([3])

    cdf = lambda x: multivariate_normal.cdf(x, mean=mu, cov=Sigma)

    print(dprobND(x, cdf))
    print(cdf(x + 1) - cdf(x))
    print(dprob_bin(x, cdf))


    # print('5 Dim test')
    # mu = [5, 4, 7, 5, 2]
    # Sigma = [[9, 0, 0, 0, 0], [0, 9, 0, 0, 0], [0, 0, 9, 0, 0], [0, 0, 0, 9, 0], [0, 0, 0, 0, 9]]
    #
    # se = test(mu, Sigma, 5000, dprobND)
    #
    # print('Test MSE = ')
    # print(np.mean(se))

    print('2 Dim test')
    mu = [5, 3]
    Sigma = [[9, 0], [0, 9]]

    se = test(mu, Sigma, 1000, dprobND)

    print('Test MSE = ')
    print(np.mean(se))

    print('3 Dim test')
    mu = [5, 3, 7]
    Sigma = [[9, 0, 0], [0, 9, 0], [0, 0, 9]]
    se = test(mu, Sigma, 1000, dprobND)
    print('Test MSE = ')
    print(np.mean(se))

    print('4 Dim test')
    mu = [5, 3, 7, 2]
    Sigma = [[9, 0, 0, 0], [0, 9, 0, 0], [0, 0, 9, 0], [0, 0, 0, 5]]
    se = test(mu, Sigma, 1000, dprobND)
    print('Test MSE = ')
    print(np.mean(se))

    print('5 Dim test')
    mu = [5, 2, 3, 7, 2]
    Sigma = [[9, 0, 0, 0, 0], [0, 9, 0, 0, 0], [0, 0, 9, 0, 0],
             [0, 0, 0, 5, 0], [0, 0, 0, 0, 18]]
    se = test(mu, Sigma, 1000, dprobND)
    print('Test MSE = ')
    print(np.mean(se))