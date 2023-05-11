from unittest import registerResult
from cut_src import cut_tfim
import numpy as np

import pickle 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--qubits", type=int, help="Number of qubits", default=10)
parser.add_argument("--random_layers", type=int,help="layers of random circuit", default=5)
parser.add_argument("--cb_layers", type=int,help="layers of CB reductor", default=2)
parser.add_argument("--epsilon", type=int, help="Probability of getting a non-considered coefficient in the cut", default=10)
parser.add_argument("--seed", type=int, help="seed for random", default=1)

import os




def main(qubits, random_layers, cb_layers, epsilon, seed):
    print('QUBITS', qubits)
    print('INPUT LAYERS', random_layers)
    print('RED LAYERS', cb_layers)
    print('EPSILON', epsilon)
    print('SEED', seed)
    np.random.seed(seed)
    
    rm = cut_tfim(qubits, random_layers, cb_layers, epsilon=epsilon/100)
    
    max_steps = 500  
    target_params = 2 * np.pi * np.random.rand(len(rm.params1))

    results = []
    rm.reductor_params = np.zeros_like(rm.reductor_params)
    for res in rm.initialize_reductor(target_params, max_steps, method='cma', options={'maxfevals':np.inf, 'maxiter':250}):
        res['params1'] = rm.params1
        res['params2'] = rm.params2
        results.append(res)

    results[-1]['target_params'] = target_params

    if 'status' not in results[-1].keys():
        folder = '/marisdata/perezsalinas/depth-limit2/results_param_init_fixM/initialize_tfim'

        os.makedirs(folder, exist_ok=True)

        np.savetxt(folder + '/%sQ_%sr_%sCB_%seps_%s.txt'%(qubits, random_layers, cb_layers, epsilon, seed), np.array(rm.initialize_reductor_hist))

        with open(folder + "/%sQ_%sr_%sCB_%seps_%s.json"%(qubits, random_layers, cb_layers, epsilon, seed), "wb") as f:
            pickle.dump(results, f)



if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(**args)
