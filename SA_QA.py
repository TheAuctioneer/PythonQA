import numpy as np
import scipy.io as sio

# import dwave library for qubo
import dimod
# import dwave solver and embedding composer
from dwave.system import DWaveSampler, composites 
# import of the qubo/ising solvers
from qubovert.sim import anneal_qubo
#import inspector to check embedding properties
import dwave.inspector
#import the brute force solver
from qubovert.utils import solve_qubo_bruteforce
# import of qubovert library
import qubovert as qv


 # Dwave account token
token_file_path = 'data/api_token.txt'
token = None

try:
    with open(token_file_path, 'r') as file:
        token = file.read().strip()
except FileNotFoundError:
    print("API token file not found.")

# Import Q matrix
importedQ = sio.loadmat('data/Q130023003300.mat')
Q_matlab = importedQ['Q']

n,m = Q_matlab.shape
qubo = {}

for i in range(n):
    for j in range(i, n):
        if Q_matlab[i,j] != 0.0:
            qubo[(i, j)] = Q_matlab[i, j]
             

#### D-WAVE QUANTUM ANNEALER #####

solver = composites.LazyFixedEmbeddingComposite(DWaveSampler(token=token))
response = solver.sample_qubo(qubo, 
                              num_reads=3500, 
                              annealing_time=20, 
                              label='UNESCO TEST - 128'
                              )


# Params
# chain_strength=50
# num_reads=3500
# annealing_time=20
# anneal_offsets=
# anneal_schedule=
# answer_mode=
# auto_scale=
# flux_biases=
# flux_drift_compensation=
# h_gain_schedule=
# initial_state=
# max_answers=
# num_spin_reversal_transforms=
# programming_thermalization=
# readout_thermalization=
# reduce_intersample_correlation=
# reinitialize_state=
# time_limit=


bestSolution = response.first.sample
result = {"xopt": list(bestSolution.values())}
# sio.savemat('QA_QDWAVE_results.mat', result)

print(result)
# print(response.info["embedding_context"]["chain_strength"])   
# chains = response.info["embedding_context"]["embedding"].values()  
# print(max(len(chain) for chain in chains))
# print("Percentage of samples with >10% breaks is {} and >0 is {}.".format(
#       np.count_nonzero(response.record.chain_break_fraction > 0.10)/3500*100,
#       np.count_nonzero(response.record.chain_break_fraction > 0.0)/3500*100)) 

       
##### Simulated annealing of qubovert, repeted 100 times, with an annealing duration of 100000 #####

# anneal_res = qv.sim.anneal_qubo(qubo, num_anneals=100, anneal_duration=1000)
# print("Energy " + format(anneal_res.best.value) + "\n")
# print("Solution " + format(anneal_res.best.state) + "\n")

# result = {"profit": anneal_res.best.value, "xopt": list(anneal_res.best.state.values())}
# sio.savemat('SA_Q11003100_results.mat', result)