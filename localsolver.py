import numpy as np
import scipy as sp
import random
import math
import csv
import ast
import os
import argparse
import sys
import objectives
import PointsGL
import matlab.engine

# from pdfo import pdfo

from skquant.opt import minimize

# import matlab.engine


def LSS(x, leng):
    from scipy import optimize

    eta = 0.05 / leng
    # gra = braningrad(x)
    # gra = shekelgrad(x)
    gra = optimize.approx_fprime(x, getattr(objectives, Problem), epsilon=0.1)
    return x - eta * gra


def LSS2(x, leng):
    from scipy import optimize

    eta = 0.5
    ep = 0.1
    # gra = braningrad(x)#
    gra = shekelgrad(x)
    # gra=optimize.approx_fprime(x,shekel,epsilon=ep)
    # gra = (gra + shekelgrad(x))/2
    return (x * leng + (x - eta * gra)) / (leng + 1)  # Polyak averaging


## To call ASTRO-DF from python
def ASTRODF(x, leng, Problem, bounds, eng):
    Result = eng.RunWrapper2([Problem], ["ASTRDFB"], matlab.double(x.tolist()), matlab.int64([1]), [str(leng)], nargout=3)
    sort = np.flip(np.argsort(np.transpose(np.array(Result[1])[1:]))) + 1
    Output = {"Points": np.array(Result[0])[sort][0], "Values": (np.array(Result[1])[sort][0]), "Counts": (np.array(Result[2])[sort][0])}
    return Output


def Bobyqa(x, leng, Problem, bounds):
    PointsGL.init()
    Result, history = minimize(getattr(objectives, Problem), x, bounds, leng, method="PyBobyqa")
    history = np.array(PointsGL.Points)
    sort = np.flip(np.argsort(np.transpose(history)[0]))
    Output = {"Points": history[:, np.linspace(1, len(x), len(x)).astype(int)][sort], "Values": history[:, 0][sort], "Counts": [[1]] * len(history[:, 0][sort])}
    # hist=np.array(PointsGL.Points)
    # sort1=np.flip(np.argsort(np.transpose(hist)[0]))
    # point=hist[:,np.linspace(1,len(x),len(x)).astype(int)][sort1]
    # print(hist,history)
    return Output


def Snobfit(x, leng, Problem, bounds):
    PointsGL.init()
    Result, history = minimize(getattr(objectives, Problem), x, bounds, leng, method="SnobFit")
    history = np.array(PointsGL.Points)
    sort = np.flip(np.argsort(np.transpose(history)[0]))
    Output = {"Points": history[:, np.linspace(1, len(x), len(x)).astype(int)][sort], "Values": history[:, 0][sort], "Counts": [[1]] * len(history[:, 0][sort])}
    return Output


def PDFO(x, leng, Problem, bounds):
    PointsGL.init()
    Result = pdfo(getattr(objectives, Problem), x, bounds=bounds, options={"maxfev": leng})
    history = np.array(PointsGL.Points)
    sort = np.flip(np.argsort(np.transpose(history)[0]))
    Output = {"Points": history[:, np.linspace(1, len(x), len(x)).astype(int)][sort], "Values": history[:, 0][sort]}
    return Output
