from cut_src import cut_model
import numpy as np

import pickle 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--qubits", type=int, help="Number of qubits", default=8)
parser.add_argument("--random_layers", type=int,help="layers of random circuit", default=5)
parser.add_argument("--cb_layers", type=int,help="layers of CB reductor", default=2)
parser.add_argument("--epsilon", type=int, help="Probability of getting a non-considered coefficient in the cut", default=1)
parser.add_argument("--ansatz", type=str, help="Considered input circuit", default='pair')
parser.add_argument("--seed", type=int, help="seed for random", default=0)

import os


def main(qubits, random_layers, cb_layers, epsilon, ansatz, seed):
    rm = cut_model(qubits, random_layers, cb_layers, epsilon=epsilon / 100, ansatz=ansatz)
    np.random.seed(seed)
    rm.update_t(1)



    results=rm.optimize_reductor(method='cma', options={'maxfevals':np.inf, 'maxiter':500}, gradual_activation=False, sigma=1)
    results['params1'] = rm.params1
    results['params2'] = rm.params2
    
    folder = '../results_soft_init/random_red/' + ansatz

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
