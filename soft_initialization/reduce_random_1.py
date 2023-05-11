from cut_src import Random_model
import numpy as np

import pickle 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--qubits", type=int, help="Number of qubits", default=8)
parser.add_argument("--random_layers", type=int,help="layers of random circuit", default=4)
parser.add_argument("--cb_layers", type=int,help="layers of CB reductor", default=3)
parser.add_argument("--epsilon", type=int, help="Probability of getting a non-considered coefficient in the cut, in log scale", default=2)
parser.add_argument("--seed", type=int, help="seed for random", default=0)
#parser.add_argument("--phase", action='store_true', help="use phases in the reductor")

import os




def main(qubits, random_layers, cb_layers, epsilon, seed, phase=True):
    rm = Random_model(qubits, random_layers, cb_layers, epsilon=10**(-epsilon), phase=True)
    np.random.seed(seed)

    rm.params1 = 2 * np.pi * np.random.rand(rm.num_params1)
    rm.params2 = 2 * np.pi * np.random.rand(rm.num_params2)

    #rm.reductor_params = np.concatenate((np.zeros(3 * rm._qubits), -np.flip(rm.params1)[:(len(rm.reductor_params) - 3 * rm._qubits)]))
    rm.reductor_params += 2 * np.pi * np.random.rand(len(rm.reductor_params))

    results=rm.optimize_reductor(method='l-bfgs-b', options={'maxfun':np.inf})
    results['params1'] = rm.params1
    results['params2'] = rm.params2

    print(results)
    
    folder = 'results/random'
    if phase: 
        folder += '_phase'

    try:
        os.makedirs(folder)
    except:
        pass

    np.savetxt(folder + '/%sQ_%sr_%sCB_%seps_%s.txt'%(qubits, random_layers, cb_layers, epsilon, seed), np.array(rm.reductor_hist))


    with open(folder + "/%sQ_%sr_%sCB_%seps_%s.json"%(qubits, random_layers, cb_layers, epsilon, seed), "wb") as f:
            pickle.dump(results, f)



if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(**args)
