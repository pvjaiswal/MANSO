# MANSO
Multistart Algorithm for Identifying All Optima of Nonconvex Stochastic Functions

MANSO is a multistart algorithm to identify all local minima
 of a constrained, nonconvex stochastic optimization problem. The algorithm
 uniformly samples points in the domain and then starts a local stochastic
 optimization run from any point that is the ``probabilistically best'' point in
 its neighborhood.
 

## Dependencies
1. Create a python 3.7 environment 

		conda create --name py37 python=3.7
		source activate py37

2. Install the following 
	Numpy,Scipy,qiskit(if run QAOA), scikit-quant(Snobfit and Bobyqa)
	
		pip install numpy
		pip install scipy
		pip install qiskit
		python -m pip install scikit-quant

3. Require MATLAB engine to use ASTRODF (https://github.com/simopt-admin/simopt/wiki) from Python (<= 3.7)

	(Tested on MATLABR2020a)
	
	https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
		

## Run MANSO from the command prompt

1. Help: python MANSO.py --help

2. Example: To find local optima of Branin function (https://www.sfu.ca/~ssurjano/branin.html) use

		python MANSO.py -b 0.1 -t 0.01 -o 0.01 -n 5 -B 15000 -OB 500 -d 2 -P Branin -seed 12 -lso ASTRODF

## To plot data profiles use Performance_Test.ipynb



