import numpy as np
import scipy as sp
import random
import math
import csv
import ast
import os
import argparse
import localsolver
import objectives
import PointsGL
import matlab.engine



'''

To install dependencies
1. Create a python 3.7 environment 

		conda create --name py37 python=3.7
		source activate py37

2. Install the following 
	
	Numpy and Scipy
	
		pip install numpy
		pip install scipy
	
    Qiskit for QAOA:
    	pip install qiskit

	scikit-quant ﻿for Snobfit and Bobyqa
    	python -m pip install scikit-quant ﻿

	pdfo 
		pip install pdfo
	

3. Require MATLAB engine to use ASTRODF (implemented in MATLAB) from Python (<= 3.7)

	(Tested on MATLABR2020a)
	https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
		

4. To run MANSO from command prompt

python MANSO.py -b 0.1 -t 0.01 -o 0.01 -n 5 -B 15000 -OB 500 -d 2 -P Branin -seed 12 -lso ASTRODF

'''

class SaneFormatter(argparse.RawTextHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MANSO',
                                     formatter_class=SaneFormatter)
    #parser.add_argument("-m", "--events", dest="EVENTS", type=str, required=True,
    #                    default=[], nargs='+', help="Array of number of events")
    parser.add_argument("-b", "--Beta", dest="BETA", type=float,
                        default=0.1, help="S1-Table 1-Confidence")
    parser.add_argument("-t", "--Boundary", dest="TAU", type=float,
                        default=0.01, help="Boundary tolerance ")
    parser.add_argument("-o", "--Omega", dest="OMEGA", type=float,
                        default=0.5, help="omega-identifying condition and Step 7 of MANSO")
    parser.add_argument("-n", "--No", dest="N0", type=int,
                        default=5, help="Number of MC sample")
    parser.add_argument("-B", "--Budget", dest="B", type=int,
                        default=1000, help="Available Sampling Budget")
    parser.add_argument("-OB", "--OBudget", dest="OB", type=int,
                        default=50, help="Budget after which Step 7 in MANSO is checked for each local run")
    parser.add_argument("-P","--Problem", dest="PROB", type=str, required=True,
                        default=None, help="Problem name- Shekel (d=4,6,8,10) or Branin (d=2)")
    parser.add_argument("-d", "--Dim", dest="DIM", type=int,
                        default=5, help="Dimension of the problem")
    parser.add_argument("-seed", "--Seed", dest="SEED", type=int, default=5, help="Seed")
    parser.add_argument("-lso", "--Solver", dest="LSO", type=str, default="Bobyqa", help="Snobfit or Bobyqa or ASTRODF or PDFOO")
    args = parser.parse_args()


beta=args.BETA # (S1) of Table 1
tau=args.TAU ## Tolerance for Boundary Conditions
omega=args.OMEGA # omega-identifying condition and Step 7 of MANSO
n0=args.N0 # Number of Monte carlo samples required at each sampled point
Problem= args.PROB
d= args.DIM
Path=os.getcwd()
Inp=args.SEED
LSO= args.LSO

if Problem == "Branin":
    LB=[-5,0]
    UB=[10,15]
elif Problem == "Shekel":
    print("Change 'd' in /MATLAB/Problems/Shekel/{shekel.m (ln 44) and shekelstructure.m (ln 44) } to the desired dimension")
    LB=[0]*d
    UB=[10]*d
elif Problem == "Qtum":
    print("Qtum")
    LB=[0]*d
    UB=[np.pi]*d
else:
    sys.exit("Must be one of {Branin,Shekel,Qtum}")

if LSO =="ASTRODF":
    import matlab.engine
    os.chdir(Path+"/MATLAB/Experiments")
    eng = matlab.engine.start_matlab()
    os.chdir(Path+"/MATLAB/Problems/"+Problem)
    eng1 = matlab.engine.start_matlab()
    


## MAIN ALGORITHM
def Nearoptima(x,Xstar,omega):
    for i in range(len(Xstar)):
        if dist(x,Xstar[i]) < omega:
            return True
    return False


def Nearbdry(x,LB,UB,tau):
    #x = np.array(x)
    #x[range(len(x))]+=tau
    for i in range(len(LB)):
        if x[i] < LB[i]+tau or x[i] > UB[i]-tau:
            return True
    return False

def ProbBetter(x,k,beta,S,N,funVAL,LB,UB,Problem,LSO):
    Nx = int(N[0])#int(np.max(N))
    d=len(LB)
    #print(matlab.double(x.tolist()))
    #Fun=getattr(foo,Problem)
    if LSO=="ASTRODF":
        xval=[getattr(eng1,Problem)(matlab.double(x.tolist()),1,0,1,nargout=1) for ct in range(Nx)]
    else:
        xval= [getattr(objectives,Problem)(x) for ct in range(Nx)]
    #xval=
    ## The above step is inefficient as I am sampling values again at the same point. 
    #We can always pass the values to this function  
    xval=np.array(xval,dtype="float")
    vol=np.prod(np.subtract(UB,LB))
    sigma=5
    M=len(S)
    #((1/d)**2)*
    rk= 1/(np.pi)**(0.5)* (math.gamma(1+d/2)*vol*sigma*math.log(M)/M)**(1/d)
    #print(rk)
    for i in range(M):
        mi= Nx#int(max(Nx,N[i]))
        zval = np.array(funVAL[i],dtype="float")
        dista= dist(x,S[i])
        # var is the variance bound (estimated) in condition S1 it depends on the distance between two points
        var = 1/mi * np.var(xval-zval) * 1/beta
        if  dista < rk and (np.sum(np.less(zval-xval,np.repeat(var**0.5,mi)))/mi ) > 1-beta:
            return True
    return False


def LSSconditions(x,beta,S,L,A,k,N,funVAL,LB,UB,omega,tau,Xstar,Problem,LSO):
    global v
    d=len(LB)
    if any((a == x).all() for a in A) :
        return False
    elif Nearoptima(x,Xstar,omega):
        return False
    elif Nearbdry(x,LB,UB,tau):
        return False
    elif ProbBetter(x,k,beta,S,N,funVAL,LB,UB,Problem,LSO):
        v= v+1
        return False
    else:
        return True
    
def dist(x,y):
    return np.sum(np.square(np.subtract(x,y)))**0.5


def dists(x,y):
    return ast.literal_eval(str(round(np.sum(np.square(np.subtract(x,y)))**0.5,3) ))

# To check if an iterate is close to the iterate of any other LSS sequence
def Prox(L,it,omega):
    x = L[it][len(L[it])-1]
    for i in range(len(L)):
        if i != it:
            for y in L[i]:
                if dist(x,y) < 2*omega:
                    return True
    return False

def xunif(LB,UB):
    x = runif.uniform(LB[0],UB[0],1)
    for i in range(len(LB)):
        if i !=0:
            x = np.append(x,runif.uniform(LB[i],UB[i]))
    return x


'''
# x is the new sampled point to tested for starting a local run
# beta is the user defined confidence level
# S is the set of all samples points
# A is the set of all points from which LSS is active
# N is the corresponding number of samples of the stochastic functiona at each point in S
# funVAL is a vector of samples values collected at each point in S.
# LB and UB are the end points of the decision space
'''


PointsGL.init()
MANSOP20=[]
MANSOP20N=[]
for seed in range(1):
    Budget=args.B  #MANSO counter
    seed=seed+int(Inp)
    OptimBudget=args.OB
    v=0 # Counter for Probabilistically best point
    runif=np.random.RandomState(seed)
    rnorm=np.random.RandomState(seed+1)
    Xstar=[]
    Xval=[]
    minval=500 # 500 for branin
    Reg=0 # Regret evaluated using true function
    d=len(LB) #dimension of the domain of the objective
    k=0
    m=0
    active=0
    j=0
    x1 = xunif(LB,UB)
    MANSOP=[x1]
    S=[x1]
    # Matrix of n_0 stochastic function values at each sampled point #[shekel(S[k],n0)]
    #globals()[Problem]
    if LSO=="ASTRODF":
        funVAL=[getattr(eng1,Problem)(matlab.double(S[k].tolist()),1,0,1,nargout=1) for ct in range(n0)]
    else:
        funVAL=[getattr(objectives,Problem)(S[k]) for ct in range(n0)]
    Budget-=n0
    N=[n0] # Number of montecarlo samples generated at each sampled/ visited point
    A=[S[0]] # Matrix of Active points
    count=[1]# Dummy count variable
    active=active+1
    L=[[S[k]]] # Matrix of points generated from Local stochastic searches from active points
    Mcount=0
   # '''
    while Budget>0:
        Alen=len(A)
        for it in range(Alen):
            if not any((a == 1 or a==2).all() for a in A[it]):
                leng = len(L[it])
                if LSO=="ASTRODF":
                    Output=getattr(localsolver,LSO)(L[it][leng-1],OptimBudget,Problem,np.transpose([LB,UB]),eng)
                else:
                    Output=getattr(localsolver,LSO)( L[it][leng-1],OptimBudget,Problem,np.transpose([LB,UB]))
                Result=Output['Points']
                NumEval=Output['Counts']
                #Budget-=OptimBudget
                for i in range(len(Result)):
                    L[it].append(Result[i])
                    MANSOP.append(Result[i])
                    N.append(NumEval[i][0])
                    Budget-=NumEval[i][0]
                if Prox(L,it,omega) or Nearbdry(L[it][leng],LB,UB,tau):# or ProbBetter(L[it][leng],k,beta,S,N,funVAL,d,X1B,X2B):
                    A[it]=np.array([2])
                    active=active-1
                    j=j+1 # updating counter for BC
                print("Optimal point in this active run so far is ",Output['Points'][-1], "with value",Output['Values'][-1])
        Mcount=Mcount+1
        print (Mcount)
        if Mcount > 500:
            break
        while active <= 10: # to ensure that we evaluate at most 10 sampled point at any k^th iteration.
            k = k + 1 # number of sampled point and also the number of iteration of MANSO: Note: We proceed one step of each active LSO from k to k+1.
            #print(active)
            x1 = xunif(LB,UB)
            MANSOP.append(x1)
            S.append(x1) # Appending uniformly sampled point to S.
            count.append(0)
            if LSO=="ASTRODF":
                funVAL.append([getattr(eng1,Problem)(matlab.double(S[k].tolist()),1,0,1,nargout=1) for ct in range(n0)])
            else:
                funVAL.append([getattr(objectives,Problem)(S[k]) for ct in range(n0)])
            Budget-=n0
            N.append(n0) # Appending number of stochastic samples
            for p in range(len(S)):
                if LSSconditions(S[p],beta,S,L,A,k,N,funVAL,LB,UB,omega,tau,Xstar,Problem,LSO) and count[p]==0: # checking our LSO starting conditions.
                        A.append(S[p]) # appending to active search sampled points
                        count[p]=1
                        active=active+1
                        L.append([S[p]]) # appending to LSO intial point.
                        m=m+1 # number of LSO search started
    MANSOP20.append(MANSOP)
    MANSOP20N.append(N)
    os.chdir(Path)
    dirName="Results"
    if not os.path.exists(dirName):
            os.makedirs(dirName)
    np.save("Results/"+Problem+"_"+LSO+"_seed_"+str(seed)+"_D_"+str(d)+"_b_"+str(beta)+"_omega_"+str(omega)+"_n0_"+str(n0)
            +"_Ob_"+str(OptimBudget)+"_B_"+str(args.B)+".npy",MANSOP20)
    np.save("Results/"+Problem+"_"+LSO+"_seed_"+str(seed)+"_ND_"+str(d)+"_b_"+str(beta)+"_omega_"+str(omega)+"_n0_"+str(n0)
    +"_Ob_"+str(OptimBudget)+"_B_"+str(args.B)+".npy",MANSOP20N)
if LSO =="ASTRODF":
    eng.quit()
    eng1.quit()
