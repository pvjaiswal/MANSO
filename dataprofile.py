import numpy as np
import math
from itertools import product
import matplotlib.pyplot as plt
import re
import objectives


def dist(x, y):
    return np.sum(np.square(np.subtract(x, y))) ** 0.5


def Performance_test(data, feval, dim, f, Tru_min, prob, tau=1.1, tau1=0.0001, length=10000):

    if f == "branind":
        True_minima = np.array([Tru_min])  # [[-np.pi, 12.275],[np.pi, 2.275], [9.42478, 2.475]]
        # glob_min= getattr(objectives,Branin_D)(True_minima[0])
        LB = [-5, 0]
        UB = [10, 15]
    elif f == "shekeltr":
        True_minima = np.array([Tru_min])
        # glob_min= getattr(objectives,Shekel_D)(True_minima[0])
        LB = [0] * dim
        UB = [10] * dim
    elif f == "stybtang4d":
        True_minima = np.array(list(product([-2.90, 2.74], repeat=4)))
        # glob_min=f(np.array([-2.90]*4))
        LB = [-5] * 4
        UB = [5] * 4
    elif f == "stybtang8d":
        True_minima = np.array(list(product([-2.90, 2.74], repeat=8)))
        # glob_min=f(np.array([-2.90]*8))
        LB = [-5] * 8
        UB = [5] * 8

    vol = np.prod(np.subtract(UB, LB))
    d = len(LB)
    tol = 1 / (np.pi) ** (0.5) * (math.gamma(1 + d / 2) * vol * tau1) ** (1 / d)
    print(tol)
    data_dist = []
    for da in data:
        # print(len(da))
        data_dist.append(np.transpose(np.reshape([dist(x, point) <= tol for x in True_minima for point in da], (len(True_minima), len(da)))))

    t1 = [100000000] * prob
    # alpha
    a = 2
    d = 2
    l = length
    rho_locmin = []

    for j in range(len(data_dist)):
        # if dataval[j][i]-glob_min <= (1-tau) * (funcent- glob_min):
        if min([len(np.where(np.transpose(data_dist[j])[i] == True)[0]) for i in range(len(True_minima))]) == 0:
            # ensures that if the minim is not identified uptill all the evaluations
            t1[j] = l * (len(LB) + 2)
        else:
            test = np.array([(np.where(np.transpose(data_dist[j])[i] == True)[0][0]) for i in range(len(True_minima))])
            t1[j] = max(test)
            # print(sum(test<l*(len(LB)+2)))
    print(t1)
    rho_locmin = [sum(np.array(t1) <= np.array([alpha * (len(LB) + 1)] * prob)) / prob for alpha in range(a, l, d)]
    alpha2 = np.linspace(a, l - d, len(rho_locmin))
    return [0, 0, rho_locmin, alpha2]


def randBranin(Samples, n0, d, seed, Samp_Paths):
    runif = np.random.RandomState(seed)
    LB = [-5, 0]
    UB = [10, 15]
    size = int(Samples / n0)
    # data=[[xunif(LB,UB,runif) for i in range(size)]for j in range(Samp_Paths)]
    data = [np.array([np.tile(xunif(LB, UB, runif), (n0, 1)) for i in range(size)]).reshape(size * n0, d) for j in range(Samp_Paths)]
    return data


def randShekel(Samples, n0, d, seed, Samp_Paths):
    runif = np.random.RandomState(seed)
    LB = [0] * d
    UB = [10] * d
    size = int(Samples / n0)
    # data=[[xunif(LB,UB,runif) for i in range(size)]for j in range(Samp_Paths)]
    data = [np.array([np.tile(xunif(LB, UB, runif), (n0, 1)) for i in range(size)]).reshape(size * n0, d) for j in range(Samp_Paths)]
    return data


def xunif(LB, UB, runif):
    x = runif.uniform(LB[0], UB[0], 1)
    for i in range(len(LB)):
        if i != 0:
            x = np.append(x, runif.uniform(LB[i], UB[i]))
    return x
