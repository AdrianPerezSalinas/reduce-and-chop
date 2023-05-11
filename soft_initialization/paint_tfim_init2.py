import pickle

from cut_src2 import cut_tfim

qubits = 10
random_layers = 4
epsilon = 15
cb_layers = 2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import os


def asymetric_std(a, axis=0):
    m = np.mean(a, axis=axis)
    a_plus = np.empty_like(m)
    a_minus = np.empty_like(m)
    for i in range(len(m)):
        a_p = a[:, i]
        a_p = a_p[a_p > m[i]]
        a_plus[i] = np.sqrt(np.mean(abs(a_p - m[i])**2))
        a_m = a[:, i]
        a_m = a_m[a_m > m[i]]
        a_minus[i] = np.sqrt(np.mean(abs(a_m - m[i])**2))
    return a_plus, a_minus


folder = "../results_soft_init/initialize_tfim_double"

max_entries = qubits ** 3 / 4
confident_entries = int(max_entries * 1.15)


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharey=True)

Cb_ranks1 = []
Cb_ranks2 = []

counter = 0
for file in os.listdir(folder):
    if "%sQ_%sr_%sCB_%seps"%(qubits, random_layers, cb_layers, epsilon) in file:
        #if counter == 25: break
             
        if "2.json" in file:
            file_root = file[:-6] 
            with open(folder + '/' + file, "rb") as f:
                results2 = pickle.load(f)[-1]

            with open(folder + '/' + file_root + "1.json", "rb") as f:
                results1 = pickle.load(f)[-1]

            if 'status' not in results1.keys() and 'status' not in results2.keys():
                counter += 1

                cb_ranks1 = []
                with open(folder + '/' + file_root + '1.txt', "r") as f:
                    for i, line in enumerate(f.readlines()):
                        l = line.split('\n')[0]
                        l = line.split(' ')
                        cb_ranks1.append(min(float(l[0]), confident_entries) / 2**qubits)
                Cb_ranks1.append(cb_ranks1)

                cb_ranks2 = []
                with open(folder + '/' + file_root + '2.txt', "r") as f:
                    for i, line in enumerate(f.readlines()):
                        l = line.split('\n')[0]
                        l = line.split(' ')
                        cb_ranks2.append(min(float(l[0]), confident_entries) / 2**qubits)
                Cb_ranks2.append(cb_ranks2)

lens1 = np.array([len(costs) for costs in Cb_ranks1])
ind1 = np.argsort(lens1)

lens2 = np.array([len(costs) for costs in Cb_ranks2])
ind2 = np.argsort(lens2)



lenM = max(lens1)
inM1 = np.argmax(lens1)

for i in range(len(Cb_ranks1)):
    Cb_ranks1[i] += [Cb_ranks1[i][-1]] * (lenM - len(Cb_ranks1[i]))

lenM = max(lens2)
inM2 = np.argmax(lens2)

for i in range(len(Cb_ranks2)):
    Cb_ranks2[i] += [Cb_ranks2[i][-1]] * (lenM - len(Cb_ranks2[i]))


Cb_ranks1 = np.array(Cb_ranks1)
iters1 = np.arange(len(Cb_ranks1[0]))
ax1.plot(iters1, Cb_ranks1[inM1], color='red', alpha=1, label = 'Hardest')
Cb_ranks1 = np.sort(Cb_ranks1, axis=0)


Cb_ranks2 = np.array(Cb_ranks2)
iters2 = np.arange(len(Cb_ranks2[0]))

ax2.plot(iters2, Cb_ranks2[inM2], color='red', alpha=1, label = 'Hardest')
Cb_ranks2 = np.sort(Cb_ranks2, axis=0)

ax1.plot(iters1, np.mean(Cb_ranks1, axis=0), color='black', alpha=1, label = 'Mean')
# ax1.fill_between(iters,Cb_ranks[15], Cb_ranks[85], color='red', alpha = 0.25)
ax1.fill_between(iters1,
                #np.mean(Cb_ranks, axis=0) - np.sqrt(np.var(Cb_ranks, axis=0)), np.mean(Cb_ranks, axis=0) + np.sqrt(np.var(Cb_ranks, axis=0)), 
                Cb_ranks1[0], Cb_ranks1[-1], 
                color='black', alpha = 0.25)

ax1.legend(loc=[.65, .65])

ax2.plot(iters2, np.mean(Cb_ranks2, axis=0), color='black', alpha=1, label = 'CB rank')
# ax1.fill_between(iters,Cb_ranks[15], Cb_ranks[85], color='red', alpha = 0.25)
ax2.fill_between(iters2,
                #np.mean(Cb_ranks, axis=0) - np.sqrt(np.var(Cb_ranks, axis=0)), np.mean(Cb_ranks, axis=0) + np.sqrt(np.var(Cb_ranks, axis=0)), 
                Cb_ranks2[1], Cb_ranks2[-2], 
                color='black', alpha = 0.25)


ax1.axhline(max_entries / 2**qubits, ls='--', color='black', lw=1)
ax1.axhline(confident_entries/ 2**qubits, ls='--', color='gray', lw=1)
ax1.set_title('First reduction', fontsize=13)
ax2.axhline(max_entries / 2**qubits, ls='--', color='black', lw=1)

ax2.axhline(confident_entries/ 2**qubits, ls='--', color='gray', lw=1)
ax2.set_title('Second reduction', fontsize=13)
#plt.fill_between(iters,np.mean(Cb_ranks, axis=0), np.mean(Cb_ranks, axis=0) - np.var(Cb_ranks, axis=0), alpha = 0.25)
    #plt.plot(Costs[i], label='Costs', color='black', alpha=0.25)

# plt.setp(ax1.get_xticklabels(), visible=False)
#plt.setp(ax2.get_xticklabels(), visible=False)

pos = ax1.get_position()
pos.x0 -= 0.035
pos.x1 += 0.025
pos.y0 += 0.01
pos.y1 += 0.04
ax1.set_position(pos)


pos = ax2.get_position()
pos.x0 += 0.0
pos.x1 += 0.07
pos.y0 += 0.01
pos.y1 += 0.04
ax2.set_position(pos)



ax1.set_ylabel(r'$CB_\epsilon / 2^n$', fontsize=12)
ax1.set_ylim([0, confident_entries / 2**qubits * 1.1])
ax2.set_ylim([0, confident_entries / 2**qubits * 1.1])
#ax1.legend(loc=[.7, .7], fontsize=12)
ax1.set_xlabel('Optimizer iterations', fontsize=12)
ax2.set_xlabel('Optimizer iterations', fontsize=12)

def KL_divergence(X, Y):
    div = 0
    for x, y in zip(X, Y):
        div += x * np.log10(x / y)

    return div


KLs = []
counter = 0
for file in os.listdir(folder):
    if "%sQ_%sr_%sCB_%seps"%(qubits, random_layers, cb_layers, epsilon) in file:
        if counter == 25: break
        if "2.json" in file:
            with open(folder + '/' + file, "rb") as f:
                results = pickle.load(f)

            data2 = results[-1]
            file_root = file[:-6]
            with open(folder + '/' + file_root + '1.json', "rb") as f:
                results = pickle.load(f)

            data1 = results[-1]

            if 'status' not in data1.keys() and 'status' not in data2.keys():
                counter += 1
            

                rm = cut_tfim(qubits, random_layers, cb_layers, epsilon=epsilon / 100)
                params1 = data2['params1']
                params2 = data2['params2']
                params3 = data1['params3']
                reductor_params1 = data1['x']
                reductor_params2 = data2['x']

                rm.update_params1(params1)
                rm.params2 = params2
                rm.params3 = params3
                rm.reductor_params1 = reductor_params1
                rm.reductor_params2 = reductor_params2

                state_cut = rm.execute_full(params1, params2, params3, reductor_params1, reductor_params2)[0]
                
                state_pure = rm.circuit1()
                state_pure = rm.circuit2(state_pure)
                state_pure = rm.circuit3(state_pure)

                KLs.append(KL_divergence(np.abs(state_cut)**2, np.abs(state_pure)**2))


ax1.text(1,1.01*max_entries / 2**qubits , r'Max $CB_\epsilon$ rank allowed')
ax1.text(1,1.01*confident_entries / 2**qubits , r'Max $CB_\epsilon$ rank estimable')

#ax2.text(0.1*lenM, confident_entries / 2**qubits * 1.005 , 'Counter %s'%counter)

ax2.text(.5*lenM,1.02 * max_entries / 2**qubits * 1.02 , r'$KL = $' + '%.4f'%(np.mean(KLs)) + r'$\pm$' + '%.4f'%(np.std(KLs)), 
bbox=dict(boxstyle="round",
                   fc=(.9, 0.9, 0.9),
                   ec=(.1, 0.1, 0.1),
                   ))

ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
ax2.tick_params(axis='x', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)

if counter == 25:
    fig.savefig(folder + '/%sQ_%sr_%scb_%seps.pdf'%(qubits, random_layers, cb_layers, epsilon))

'''fig.savefig('results/random_init/figures/%sQ_%sr_%seps.pdf'%(qubits, random_layers, epsilon))

ax4.hist(KLs[:, -1], bins=np.logspace(-1.75,-0.9, 15), color='black', rwidth=0.95)
ax4.set_xscale('log')
ax4.set_xlabel('KL divergence')
ax4.set_ylabel('N appearences')
figkl.savefig('results/random_init/figures/%sQ_%sr_%seps_kl.pdf'%(qubits, random_layers, epsilon))'''
