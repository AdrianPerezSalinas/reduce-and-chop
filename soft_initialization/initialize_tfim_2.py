from cut_src2 import cut_tfim
import numpy as np

import pickle 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--qubits", type=int, help="Number of qubits", default=8)
parser.add_argument("--random_layers", type=int,help="layers of random circuit", default=4)
parser.add_argument("--cb_layers", type=int,help="layers of CB reductor", default=2)
parser.add_argument("--epsilon", type=int, help="Probability of getting a non-considered coefficient in the cut", default=10)
parser.add_argument("--seed", type=int, help="seed for random", default=0)

import os




def main(qubits, random_layers, cb_layers, epsilon, seed):
    print('QUBITS', qubits)
    print('INPUT LAYERS', random_layers)
    print('RED LAYERS', cb_layers)
    print('EPSILON', epsilon)
    print('SEED', seed)
    np.random.seed(seed)    
    rm = cut_tfim(qubits, random_layers, cb_layers, epsilon=epsilon/100)
    
    target_params1 = 2 * np.pi * np.random.rand(len(rm.params1))
    target_params2 = 2 * np.pi * np.random.rand(len(rm.params2))

    results1 = []
    rm.reductor_params1 = np.zeros_like(rm.reductor_params1)
    for res in rm.initialize_reductor1(target_params1, 250, method='cma', options={'maxfevals':np.inf, 'maxiter':500}):
        res['params1'] = rm.params1
        res['params2'] = rm.params2
        res['params3'] = rm.params3
        results1.append(res)
    results1[-1]['target_params1'] = target_params1
    results1[-1]['target_params2'] = target_params2

    folder = '../results_soft_init/initialize_tfim_double'

    try:
        os.makedirs(folder)
    except:
        pass

    np.savetxt(folder + '/%sQ_%sr_%sCB_%seps_%s_1.txt'%(qubits, random_layers, cb_layers, epsilon, seed), np.array(rm.initialize_hist1))


    with open(folder + "/%sQ_%sr_%sCB_%seps_%s_1.json"%(qubits, random_layers, cb_layers, epsilon, seed), "wb") as f:
            pickle.dump(results1, f)



    if 'status' not in results1[-1].keys():

        results2 = []
        rm.reductor_params2 = np.zeros_like(rm.reductor_params2)
        for res in rm.initialize_reductor2(target_params2, 500, method='cma', options={'maxfevals':np.inf, 'maxiter':500}):
            res['params1'] = rm.params1
            res['params2'] = rm.params2
            res['params3'] = rm.params3
            results2.append(res)
        results2[-1]['target_params1'] = target_params1
        results2[-1]['target_params2'] = target_params2

        
        folder = 'results_soft_init/initialize_tfim_double'

        np.savetxt(folder + '/%sQ_%sr_%sCB_%seps_%s_2.txt'%(qubits, random_layers, cb_layers, epsilon, seed), np.array(rm.initialize_hist2))

        with open(folder + "/%sQ_%sr_%sCB_%seps_%s_2.json"%(qubits, random_layers, cb_layers, epsilon, seed), "wb") as f:
                pickle.dump(results2, f)



if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(**args)
