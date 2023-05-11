from cut_src import cut_tfim
import numpy as np

qubits = 8
random_layers = 5 
cb_layers = 2
epsilon = 10


rm = cut_tfim(qubits, random_layers, cb_layers, epsilon=epsilon/100)
rm.params1 = 2 * np.pi * np.random.rand(len(rm.params1))

reductor_params = np.random.rand(len(rm.reductor_params))

rm.execute(reductor_params)