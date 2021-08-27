#==========================================================================
# ASTRO-DF (Adaptive Sampling Trust Region Optimization - Derivative Free)
#==========================================================================
# DATE
#        Summer 2020
#
# AUTHOR - Prateek Jaiswal
#
#
#
#       Adapted from MATLAB code by Pranav Jain, Yunsoo Ha, Sara Shashaani, Kurtis Konrad.
#       REFERENCE
#        Sara Shashaani, Fatemeh S. Hashemi and Raghu Pasupathy (2018)
#		 ASTRO-DF: A Class of Adaptive Sampling Trust Region Algorithms
#        for Derivative-Free Stochastic Optimization 28(4):3145-3176
#
#==========================================================================
#
# INPUT
#        problem
#              Problem function name
#        probstructHandle
#              Problem structure function name 
#        problemRng
#              Random number generators (streams) for problems
#        solverRng
#              Random number generator (stream) for solver
#        numBudget
#              number of budgets to record, >=3 the spacing between
#              adjacent budget points should be about the same
#
#
# OUTPUT
#        Ancalls
#              An array (size = 'NumSoln' X 1) of budget expended
#        Asoln
#              An array (size = 'NumSoln' X 'dim') of solutions
#              returned by solver
#        Afn
#              An array (size = 'NumSoln' X 1) of estimates of expected
#              objective function value
#        AFnVar
#              An array of variances corresponding to
#              the objective function at A
#              Equals math.nan if solution is infeasible
#        AFnGrad
#              An array of gradient estimates at A not reported
#        AFnGardCov
#              An array of gradient covariance matrices at A not reported
#        Aconstraint
#              A vector of constraint function estimators not applicable
#        AConstraintCov
#              An array of covariance matrices corresponding to the
#              constraint function at A not applicable
#        AConstraintGrad
#              An array of constraint gradient estimators at A not
#              applicable
#        AConstraintGradCov
#              An array of covariance matrices of constraint gradient
#              estimators at A not applicable
#
#==========================================================================

import math
import numpy as np

## ASTRO-DF

def ASTRDF(probHandle, probstructHandle, problemRng, solverRng):

## Unreported
    AFnGrad = math.nan
    AFnGradCov = math.nan
    Aconstraint = math.nan
    AConstraintCov = math.nan
    AConstraintGrad = math.nan
    AConstraintGradCov = math.nan

# Separate the two solver random number streams
    solverInitialRng = solverRng{1} # RNG for finding initial solutions

# Get details of the problem and random initial solution
    RandStream.setGlobalStream(solverInitialRng)

    [minmax, dim, ~, ~, VarBds, ~, ~, x0, budgetmax, ~, ~, ~] = probstructHandle(1)
    # get an initial solution
    budgetini = min(30, max(np.floor(0.001 * budgetmax) + 1, 5))
    [xini, varini, ~, ~, ~, ~, ~, ~] = probHandle(x0, budgetini, problemRng, 1)
    budgetmaxPT = budgetmax

    # Set the  samplingrule
    type = 3 # an integer value from 1 to 4
    # See the subfunction 'Adaptive_Sampling' for more details

    Delta_max = np.linalg.norm(VarBds(:, 2)-VarBds(:, 1), inf) # maximum  acceptable trust region radius
    if Delta_max == Inf:
        Delta_max = 100

    Delta0 = .08 * Delta_max # initial trust region radius
    shrink = .5 ** np.log(dim + 1)
    expand = 1 / shrink
    radiustry = [1 shrink expand] # fraction of initial trust region to use for parameter tuning
    Delta1 = [Delta0 Delta0 * shrink Delta0 * expand]
    setupprct = .03 # the initial percentage of the budget to use for finding a good initial radius

    ptdict = []
    ptdict_par = []
    x_points = []
    callcount = []
    func_points = []
    var_points = []
    calls = 0
    funcbest = zeros(size(radiustry))
    xbest = zeros(length(radiustry), dim)
    point_precision = 7 # number of decimal places to keep for any points in ptdict

    ptdict = struct('pts', [x0], 'counts', [0], 'means', [0], 'variances', [0], 'rands', [1], 'decimal', point_precision, 'iterationNumber', [0 0 0])

    for i in rnage(len(radiustry)):
    # try to run the algorithm on setupprct of the budget at different
    # fractions    of    the    initial    suggested radius
    [callcounti, x_pointsi, func_pointsi, var_pointsi, ptdict] = ASTRDF_Internal(probHandle, problemRng,
                                                                                 solverRng, minmax, dim,
                                                                                 VarBds, x0,floor(setupprct * budgetmaxPT)
                                                                                 , type, Delta1(i), Delta_max,
                                                                                 ptdict, 1, 0)

    infoi = ptdict.info
    ptdict = rmfield(ptdict, 'info') # only     save    the    points
    x_pointsi_{i} = x_pointsi
    func_pointsi_{i} = func_pointsi
    callcounti_{i} = callcounti
    var_pointsi_{i} = var_pointsi

    calls = calls + infoi.calls # total number of calls

    if ~isempty(func_pointsi): # if the attempt had a successful iteration  use    the    last    point
        funcbest(i) = func_pointsi(end)
        xbest(i,:)=x_pointsi(end,:)
        Delta_par(i) = ptdict.PTinfo(i).Delta
    else:
        if minmax == -1: # minimzation
            funcbest(i) = Inf # no success means value was Inf
        elif minmax == 1:
            funcbest(i) = 0
        xbest(i,:)=x0
        Delta_par(i) = Delta1(i)

    # pick the best value from the trials
    funcbest = -1 * minmax * funcbest
    if minmax == -1:
        [bestval, best] = min(funcbest)
    elif minmax == 1:
        [bestval, best] = max(funcbest)

    BestS = 0

    for i in range(3):
        if best == i:
            BestS = i
            x_aft_tune = xbest(i,:)
            Delta = Delta_par(i)
            x_points_par = x_pointsi_{i}
            func_points_par = cell2mat(func_pointsi_(i))
            callcount_par = cell2mat(callcounti_(i)) + budgetini + (2 * floor(setupprct * budgetmaxPT))
            var_points_par = cell2mat(var_pointsi_(i))
            break
    # budgetmax = budgetmax - budgetini - calls
    # run the main algorithm
    [callcount, x_points, func_points, var_points] = ASTRDF_Internal(probHandle, problemRng, solverRng, minmax, dim,
                                                VarBds, x_aft_tune, budgetmaxPT - 3 * floor(setupprct * budgetmaxPT),
                                                                     type, Delta, Delta_max, ptdict, 0, BestS)

    callcount = callcount + 3 * floor(setupprct * budgetmaxPT)

    # record points for new SIMOPT format Jan 2020:
        Asoln = [x0,x_points_par,x_points]
        Afn = [xini,func_points_par,func_points]
        AFnVar = [varini,var_points_par,var_points]
        Ancalls = [budgetini,callcount_par,callcount]

    return (Ancalls, Asoln, Afn, AFnVar, AFnGrad, AFnGradCov, Aconstraint, AConstraintCov, AConstraintGrad, AConstraintGradCov)

#
# function[callcount, x_points, func_points, var_points, ptdict]...
# = ASTRDF_Internal(probHandle, problemRng, solverRng, ...
# minmax, dim, VarBds, xk, budgetmax, type, Delta0, Delta_max, ptdict, PT, BestS)
# % ASTRDF_Internal
# runs
# the
# main
# portion
# of
# the
# ASTRO - DF
# Algorithm
# %
# % INPUTS
# %
# % probHandle = the
# problem
# handle
# % problemRng = Random
# number
# generators(streams)
# for problems
#     % solverRng = Random
#     number
#     generator(stream)
#     for solver
#         % minmax = + / - 1
#     whether
#     the
#     goal is minimization or maximization
# % = 1 if problem is being
# maximized
# % =-1 if objective is being
# minimized
# % dim = the
# dimension
# of
# the
# problem
# % VarBds = dim
# x
# 2
# array
# of
# the
# lower and upper
# bounds
# for each input
#     % variable
#     dimension
# % x0 = the
# initial
# point
# that
# the
# algorithm
# uses
# % budgetmax = the
# maximum
# number
# of
# function
# calls
# allowed
# % type = an
# integer
# used
# to
# determine
# the
# adaptive
# sample
# size
# rule
# % see
# Adaptive_Sampling
# function
# for details
#     % Delta0 = the
#     initial
#     trust
#     region
#     radius
#     size
# % Delta_max = the
# maximum
# allowed
# trust
# region
# radius
# % ptdict = (optional)
# the
# data
# dictionary
# structure
# that
# keeps
# track
# of
# visited
# points
# %.pts = the
# data
# points
# that
# have
# been
# visited
# %.counts = the
# number
# of
# times
# the
# function
# has
# been
# called
# at
# % that
# point
# %.means = the
# mean
# values
# that
# the
# function
# has
# been
# called
# at
# a
# % point
# %.variances = the
# variance
# values
# of
# the
# function
# at
# a
# point
# %.rands = the
# next
# random
# seed
# to
# use
# at
# a
# point
# %.decimal = the
# number
# of
# places
# after
# the
# decimal
# point
# to
# % round
# values
# off
# % (optional inclusions for warmstarting)
# %.x_points = the
# visited
# incumbent
# solutions
# %.callcount = the
# number
# of
# calls
# at
# the
# incumbent
# solutions
# %.func_points = the
# mean
# function
# values
# at
# the
# incumbent
# solutions
# %.var_points = the
# variances
# at
# the
# incumbent
# solutions
# %.info = information
# structure
# with fields:
#     %.calls = the
#     number
#     of
#     calls
#     already
#     made
# %.iteration_number = the
# number
# of
# iterations
# already
# made
# % PT = the
# binary
# variable
# % if PT = 0, main run
# % if PT = 1, run for parameter tuning
# %
# %
# % OUTPUTS
# %
# % callcount = an
# array
# of
# the
# number
# of
# calls
# needed
# to
# reach
# the
# % incumbent
# solutions
# % x_points = an
# array
# of
# the
# incumbent
# solutions
# % func_points = the
# estimated
# function
# value
# at
# the
# incumbent
# solutions
# % var_points = the
# estimated
# variance
# at
# the
# incumbent
# solutions
# % ptdict = a
# data
# dictionary
# structure.It
# contains
# the
# same
# information
# % as before
# with these additional fields:
#     %.info.delta = the
#     current
#     radius
# %.info.gradnorm = the
# norm
# of
# the
# current
# gradient
# %.sensitivity = the
# sensitivity
# of
# the
# variable
# bounds
#
# x_init = xk
# % Separate
# the
# two
# solver
# random
# number
# streams
# solverInitialRng = solverRng
# {1} % RNG
# for finding initial solutions
# solverInternalRng = solverRng
# {2} % RNG
# for the solver's internal randomness
#
# % Generate
# new
# starting
# point
# x0(it
# must
# be
# a
# row
# vector)
# RandStream.setGlobalStream(solverInitialRng)
#
# % More
# default
# values
# eta_1 = 0.10 % threshhold
# for decent success
#     eta_2 = 0.50 % threshhold
#     for good success
#         w = 0.99
# mu = 1.05
# beta = 1 / mu
# gamma_1 = (1.25) ^ (2 / dim) % successful
# step
# radius
# increase
# gamma_2 = 1 / gamma_1 % unsuccessful
# step
# radius
# decrease
#
# % create
# the
# output
# variables or load
# them if available
# % if nargin < 12 | | ~isfield(ptdict, 'info')
#
# x_points = []
# callcount = []
# func_points = []
# var_points = []
# % Initializations
# calls = 0
#
# if PT == 1
#     iteration_number = 1
# else
#     iteration_number = ptdict.iterationNumber(BestS)
# end
#
# % Shrink
# VarBds
# to
# prevent
# floating
# errors
# % following
# STRONG
# code
# sensitivity = 10 ^ (-5) % shrinking
# scale
# for VarBds
#     VarBds(:, 1) = VarBds(:, 1) + sensitivity
# VarBds(:, 2) = VarBds(:, 2) - sensitivity
# ptdict.sensitivity = sensitivity
#
# Delta = Delta0
# while calls <= budgetmax
#     o = 100
#     if Delta > Delta_max / o % if Delta > 1.2
#         lin_quad = 1 % run
#         a
#         linear
#         model
#     else
#         lin_quad = 2 % run
#         a
#         quadratic
#         model
#     end
#
#     % run
#     the
#     adaptive
#     sampling
#     part
#     of
#     the
#     algorithm
#     [q, Fbar, Fvar, Deltak, calls, ptdict, budgetmax] = Model_Construction(probHandle, xk, Delta, iteration_number, ...
#     type, w, mu, beta, calls, solverInternalRng, problemRng, minmax, ptdict, VarBds, lin_quad, budgetmax, PT, BestS)
#
#     % Record
#     Fbar
#     x_incumbent = xk
#     Fbar_incumbent = Fbar(1)
#     Fvar_incumbent = Fvar(1)
#
#     % Step
#     3
#     % Minimize
#     the
#     constrained
#     model
#     fun =
#
#
#     @(x)
#
#
#     Model_Approximation(x - xk, lin_quad, q)
#     nonlcon =
#
#
#     @(x)
#
#
#     disk(x, xk, Deltak)
#     [~, ~, H, ~, ~] = fun(xk)
#
#     hessint =
#
#
#     @(x, lambda ) hessinterior(x, lambda, H)
#
#
#     options.HessianFcn = hessint
#     options.OptimalityTolerance = Fvar_incumbent * 0.1
#
#     A = []
#     b = []
#     Aeq = []
#     beq = []
#     lb = max(xk - Deltak, VarBds(:, 1)')
#     ub = min(xk + Deltak, VarBds(:, 2)')
#     maxfuncevals_fmincon = 1000
#     C_Tol = 1e-12
#     S_Tol = 1e-20
#
#     options = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp', ...
#     'SpecifyObjectiveGradient', true, 'SpecifyConstraintGradient', true, ...
#     'MaxFunctionEvaluations', maxfuncevals_fmincon, 'ConstraintTolerance', C_Tol, ...
#     'StepTolerance', S_Tol)
#
#     [x_tilde, ~, exitflag, ~] = fmincon(fun, xk, A, b, Aeq, beq, lb, ub, nonlcon, options)
#
#     % Step
#     4
#     % load in the
#     current
#     point
#     's information if it exists already in the
#     % point
#     dictionary
#     [Result, LocResult] = ismember(round(x_tilde, ptdict.decimal), ptdict.pts, 'rows')
#     if Result == true
#     Fb = ptdict.means(LocResult)
#     sig2 = ptdict.variances(LocResult)
#     samplecounts = ptdict.counts(LocResult)
#     problemseed = ptdict.rands(LocResult) % set to global random seed for use in the function
#     if PT == 1
#     budgetmax = budgetmax - samplecounts
#     end
#     else
#     Fb = 0
#     sig2 = 0
#     samplecounts = 0
#     % set to global random seed for use in the function
#     % Using CRN: for
#     each
#     solution, start
#     at
#     substream
#     1
#     for problemRng
#     problemseed = 1 % Reset seed to get CRN across solutions
#     end
#
#     % sample the point enough times
#     while 1
#         if samplecounts >= Adaptive_Sampling(iteration_number, sig2, Deltak, type)
#             break
#         else
#             if samplecounts > 2 & & calls > budgetmax
#                 break
#             end
#             samplecounts = samplecounts + 1
#         end
#         [xi, ~, ~, ~, ~, ~, ~, ~] = probHandle(x_tilde, 1, problemRng, problemseed)
#         xi = -minmax * xi % Account
#         for minimization / maximization
#             problemseed = problemseed + 1 % iid
#             observations
#         calls = calls + 1
#         F_old = Fb
#         Fb = (samplecounts - 1) / samplecounts * Fb + 1 / samplecounts * xi % update
#         mean
#         sig2 = (samplecounts - 2) / (samplecounts - 1) * sig2 + samplecounts * (Fb - F_old) ^ 2 % update
#         variance
#         if samplecounts == 1
#             sig2 = 0
#         end
#     end
#
#     if calls > budgetmax
#         if PT == 1
#             if Delta0 == .08 * Delta_max
#                 ptdict.iterationNumber(1) = iteration_number
#                 ptdict.PTinfo(1).Delta = Delta
#             elseif
#             Delta0 < .08 * Delta_max
#             ptdict.iterationNumber(2) = iteration_number
#             ptdict.PTinfo(2).Delta = Delta
#         else
#             ptdict.iterationNumber(3) = iteration_number
#             ptdict.PTinfo(3).Delta = Delta
#     end
#     end
# end
#
# Fbar_tilde = Fb
# Fvar_tilde = sig2
#
# % save
# the
# information
# to
# the
# point
# dictionary
# if Result == false
#     ptdict.pts = [ptdict.pts
#     round(x_tilde, ptdict.decimal)]
#     ptdict.means = [ptdict.means
#     Fb]
#     ptdict.counts = [ptdict.counts
#     samplecounts]
#     ptdict.variances = [ptdict.variances
#     sig2]
#     ptdict.rands = [ptdict.rands
#     problemseed]
#     else
#     ptdict.means(LocResult) = Fb
#     ptdict.variances(LocResult) = sig2
#     ptdict.counts(LocResult) = samplecounts
#     ptdict.rands(LocResult) = problemseed
# end
#
# if Fbar_tilde > min(Fbar)
#     Fbar_tilde = min(Fbar)
#     x_tilde = ptdict.pts(Fbar_tilde == ptdict.means,:)
#     end
#
#     % Step
#     5 - Model
#     Accuracy
#     rho = (Fbar(1) - Fbar_tilde) / (
#                 Model_Approximation(xk - xk, lin_quad, q) - Model_Approximation(x_tilde - xk, lin_quad, q))
#
#     % Step
#     6 - Trust
#     Region
#     Update
#     step
#     if rho >= eta_2 % really good accuracy
#     xk = x_tilde
#     Delta = min(gamma_1 * Deltak, Delta_max) % expand
#     trust
#     region
#     x_points = [x_points
#     x_tilde]
#     callcount = [callcount
#     calls]
#     func_points = [func_points
#     Fbar_tilde]
#     var_points = [var_points
#     Fvar_tilde]
#     elseif
#     rho >= eta_1 % good
#     accuracy
#     xk = x_tilde
#     Delta = min(Deltak, Delta_max) % maintain
#     same
#     trust
#     region
#     size
#     x_points = [x_points
#     x_tilde]
#     callcount = [callcount
#     calls]
#     func_points = [func_points
#     Fbar_tilde]
#     var_points = [var_points
#     Fvar_tilde]
#     else % poor
#     accuracy
#     Delta = min(gamma_2 * Deltak, Delta_max) % shrink
#     trust
#     region
# end
#
# [~, currentgrad] = Model_Approximation(xk - xk, lin_quad, q)
# iteration_number = iteration_number + 1
# end
#
# % save
# final
# information
# before
# exiting
# if isempty(x_points) == false
#     x_points = [x_points
#     x_points(end,:)]
#     callcount = [callcount
#     calls]
#     func_points = [func_points
#     func_points(end)]
#     var_points = [var_points
#     var_points(end)]
#     end
#     info.iteration_number = iteration_number
#     info.delta = Delta
#     info.calls = calls
#     info.gradnorm = norm(currentgrad)
#     ptdict.info = info
# end