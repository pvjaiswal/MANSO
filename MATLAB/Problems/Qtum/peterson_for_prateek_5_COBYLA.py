import numpy as np
import pickle
from itertools import product
from functools import partial
from scipy.optimize import minimize
from qiskit import QuantumCircuit, execute, Aer
import argparse
import csv


def append_zz_term(qc, q1, q2, gamma):
    qc.cx(q1, q2)
    qc.rz(2 * gamma, q2)
    qc.cx(q1, q2)


def get_cost_operator_circuit(elist, nnodes, gamma):
    qc = QuantumCircuit(nnodes, nnodes)
    for u, v in elist:
        append_zz_term(qc, u, v, gamma)  # here can instead append the optimized C_phi gate
    return qc


def append_x_term(qc, q1, beta):
    qc.rx(2 * beta, q1)


def get_mixer_operator_circuit(nqubits, beta):
    qc = QuantumCircuit(nqubits, nnodes)
    for n in range(nqubits):
        append_x_term(qc, n, beta)
    return qc


def get_qaoa_circuit(elist, nnodes, beta, gamma, measure=True):
    assert len(beta) == len(gamma)
    p = len(beta)  # infering number of QAOA steps from the parameters passed
    qc = QuantumCircuit(nnodes, nnodes)
    # first, apply a layer of Hadamards
    qc.h(range(nnodes))
    # second, apply p alternating operators
    for i in range(p):
        qc += get_cost_operator_circuit(elist, nnodes, gamma[i])
        qc += get_mixer_operator_circuit(nnodes, beta[i])
    if measure:
        qc.barrier(range(nnodes))
        qc.measure(range(nnodes), range(nnodes))
    return qc


#
# Maxcut objective
#


def maxcut_obj(x, elist):
    cut = 0
    for i, j in elist:
        if x[i] != x[j]:
            # the edge is cut
            cut -= 1
    return cut


def brute_force(obj_f, num_variables):
    best_cost_brute = 0
    for b in range(2 ** num_variables):
        x = [int(t) for t in reversed(list(bin(b)[2:].zfill(num_variables)))]
        try:
            cost = obj_f(x)
        except TypeError:
            cost = obj_f(np.array(x))
        if cost < best_cost_brute:
            best_cost_brute = cost
            xbest_brute = x
    return best_cost_brute, xbest_brute


#
# Utils for statevector and searching for optimal parameters
#


def state_num2str(basis_state_as_num, nqubits):
    return "{0:b}".format(basis_state_as_num).zfill(nqubits)


def state_str2num(basis_state_as_str):
    return int(basis_state_as_str, 2)


def state_reverse(basis_state_as_num, nqubits):
    basis_state_as_str = state_num2str(basis_state_as_num, nqubits)
    new_str = basis_state_as_str[::-1]
    return state_str2num(new_str)


def get_adjusted_state(state):
    nqubits = np.log2(state.shape[0])
    if nqubits % 1:
        raise ValueError("Input vector is not a valid statevector for qubits.")
    nqubits = int(nqubits)

    adjusted_state = np.zeros(2 ** nqubits, dtype=complex)
    for basis_state in range(2 ** nqubits):
        adjusted_state[state_reverse(basis_state, nqubits)] = state[basis_state]
    return adjusted_state


def state_to_ampl_counts(vec, eps=1e-15):
    """Converts a statevector to a dictionary
    of bitstrings and corresponding amplitudes
    """
    qubit_dims = np.log2(vec.shape[0])
    if qubit_dims % 1:
        raise ValueError("Input vector is not a valid statevector for qubits.")
    qubit_dims = int(qubit_dims)
    counts = {}
    str_format = "0{}b".format(qubit_dims)
    for kk in range(vec.shape[0]):
        val = vec[kk]
        if val.real ** 2 + val.imag ** 2 > eps:
            counts[format(kk, str_format)] = val
    return counts


def invert_counts(counts):
    return {k[::-1]: v for k, v in counts.items()}


def compute_maxcut_energy(counts, elist):
    energy = 0
    total_counts = 0
    for meas, meas_count in counts.items():
        obj_for_meas = maxcut_obj(meas, elist)
        energy += obj_for_meas * meas_count
        total_counts += meas_count
    return energy / total_counts


def compute_maxcut_energy_sv(sv, elist):
    """Compute objective from statevector
    For large number of qubits, this is slow.
    """
    counts = state_to_ampl_counts(sv)
    return sum(maxcut_obj(np.array([int(x) for x in k]), elist) * (np.abs(v) ** 2) for k, v in counts.items())


def get_black_box_objective_sv(elist, nnodes, p):
    backend = Aer.get_backend("statevector_simulator")

    def f(theta):
        # let's assume first half is betas, second half is gammas
        beta = theta[:p]
        gamma = theta[p:]
        qc = get_qaoa_circuit(elist, nnodes, beta, gamma, measure=False)
        sv = execute(qc, backend).result().get_statevector()
        # return the energy
        return compute_maxcut_energy_sv(get_adjusted_state(sv), elist)

    return f


def get_black_box_objective(elist, nnodes, p):
    backend = Aer.get_backend("qasm_simulator")

    def f(theta):
        # let's assume first half is betas, second half is gammas
        beta = theta[:p]
        gamma = theta[p:]
        qc = get_qaoa_circuit(elist, nnodes, beta, gamma)
        counts = execute(qc, backend).result().get_counts()
        # return the energy
        return compute_maxcut_energy(invert_counts(counts), elist)

    return f


elist = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 5], [1, 6], [2, 7], [3, 8], [4, 9], [5, 7], [5, 8], [6, 8], [6, 9], [7, 9]]
nnodes = 10

obj_f = partial(maxcut_obj, elist=elist)
true_opt = brute_force(obj_f, nnodes)
# print(f"True optimum: ", true_opt)

# p is the number of QAOA alternating operators
p = 5
obj_sv = get_black_box_objective_sv(elist, nnodes, p)
obj = get_black_box_objective(elist, nnodes, p)


## I am using point gerenerted from MANSO instead of a grid.
path = "./"
Quant = ["Qtum_seed_12_10D_b1_0.1_omega_0.05_n0_1_Ob_200_B_50000.npy"]  ## There are around 17000 points in this file

Quantum = np.load(path + Quant[0], allow_pickle=True)
minimu = 100
for i, d in enumerate(Quantum[0][10356:10500]):
    if i % 40 == 0:  ## status
        print(i)
    res_sample = minimize(obj_sv, np.hstack(d), method="COBYLA", options={"maxiter": 1500, "disp": False})
    filename = "COBYLA_on_Peterson" + ".csv"
    with open(filename, "a", newline="") as file1:
        writer = csv.writer(file1)
        writer.writerow([i, d, res_sample["x"], res_sample["fun"]])
    if res_sample["fun"] < minimu:
        print("Minimum  value so far is ", res_sample["fun"], "at point", i)
        minimu = res_sample["fun"]
