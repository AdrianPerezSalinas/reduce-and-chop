import pickle

from cut_src import cut_tfim

qubits = 10
random_layers = 5
epsilon = 1
cb_layers = 2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import os

fig, (ax1, ax2) = plt.subplots(figsize=(5, 6.75), nrows=2, sharex=False)

Cb_ranks = []
Confidences = []
max_entries = qubits ** 3 / 5
confident_entries = qubits ** 3 / 4


    
counter = 0
folder = 'depth-limit2/results_param_init_fixM/initialize_tfim'
for file in os.listdir(folder):
    print(file)
    if counter == 40: break
    if "%sQ_%sr_%sCB_%seps"%(qubits, random_layers, cb_layers, epsilon) in file:
        
        if ".json" in file:

            with open(folder + '/' + file, "rb") as f:
                result = pickle.load(f)[-1]
                

        elif ".txt" in file:
            cb_ranks = []
            confidences = []
            with open(folder + '/' + file, "r") as f:
                for i, line in enumerate(f.readlines()):
                    l = line.split('\n')[0]
                    l = line.split(' ')
                    cb_ranks.append(float(l[0]) / 2**qubits)
                    confidences.append(float(l[1]))
            if max(confidences)<1e-4:
                counter += 1
                Cb_ranks.append(cb_ranks)
                Confidences.append(confidences)
print(counter)
cmap = cm.get_cmap('viridis')
num = 2
lens = np.array([len(costs) for costs in Cb_ranks])
ind = np.argsort(lens)


lenM = max(lens)
inM = np.argmax(lens)
for i in range(len(Cb_ranks)):
    Cb_ranks[i] += [Cb_ranks[i][-1]] * (lenM - len(Cb_ranks[i]))
    Confidences[i] += [Confidences[i][-1]] * (lenM - len(Confidences[i]))


Cb_ranks = np.array(Cb_ranks)
Confidences = np.array(Confidences)
iters = np.arange(len(Cb_ranks[0]))

for cb_ranks in Cb_ranks:
    ax1.plot(iters, cb_ranks, color='black', alpha=.1, lw=1)


ax1.plot(iters, Cb_ranks[inM], color='red', alpha=1, label = 'Latest convergence')


Cb_ranks = np.sort(Cb_ranks, axis=0)
Confidences = np.sort(Confidences, axis=0)

ax1.plot(iters, np.mean(Cb_ranks, axis=0), color='black', alpha=1, label = 'Average')
# ax1.fill_between(iters,Cb_ranks[15], Cb_ranks[85], color='red', alpha = 0.25)





ax1.axhline(max_entries / 2**qubits, ls='--', color='black', lw=1, label = 'Stopping CB')
ax1.axhline(confident_entries/ 2**qubits, ls='--', color='gray', lw=1, label = r'$\epsilon^2$ Measurements')
#plt.fill_between(iters,np.mean(Cb_ranks, axis=0), np.mean(Cb_ranks, axis=0) - np.var(Cb_ranks, axis=0), alpha = 0.25)
    #plt.plot(Costs[i], label='Costs', color='black', alpha=0.25)

# plt.setp(ax1.get_xticklabels(), visible=False)
#plt.setp(ax2.get_xticklabels(), visible=False)

pos = ax1.get_position()
pos.x0 += 0.05
pos.x1 += 0.05
pos.y0 -= 0.125
pos.y1 += 0.00
ax1.set_position(pos)

pos = ax2.get_position()
pos.x0 += 0.05
pos.x1 += 0.05
pos.y0 -= 0.025
pos.y1 -= 0.15
ax2.set_position(pos)



ax1.set_ylabel(r'$CB_\epsilon / 2^n$', fontsize=15)
ax2.set_ylabel('Appearances', fontsize=15)
#ax1.set_ylim([0, confident_entries * 1.1 / 2**qubits])
ax1.set_ylim([0, 1.3 * confident_entries / 2**qubits])
ax1.set_xlabel('Optimizer iterations', fontsize=15)

ax2.tick_params(axis='x', labelsize = 14)
ax1.tick_params(axis='y', labelsize = 14)
ax1.tick_params(axis='x', labelsize = 14)
ax2.tick_params(axis='y', labelsize = 14)

ax1.set_xticks(np.arange(0, lenM, lenM//5))

ax2.hist(np.array(Cb_ranks)[:, -1])
ax2.set_xlabel(r'Final $CB_\epsilon / 2^n$', fontsize=15)
leg = ax1.get_legend_handles_labels()

def KL_divergence(X, Y):
    div = 0
    for x, y in zip(X, Y):
        div += x * np.log10(x / y)

    return div


KLs = []
counter = 0
for file in os.listdir(folder):
    if counter == 25: break
    if "%sQ_%sr_%sCB_%seps"%(qubits, random_layers, cb_layers, epsilon) in file:
        
        if ".json" in file:
            counter += 1
            with open(folder + '/' + file, "rb") as f:
                results = pickle.load(f)

            data = results[-1]
            rm = cut_tfim(qubits, random_layers, cb_layers, epsilon=epsilon / 100)
            params1 = data['params1']
            params2 = data['params2']
            reductor_params = data['x']

            rm.update_params1(params1)
            rm.params2 = params2
            rm.reductor_params = reductor_params

            state_cut = rm.execute_full(params1, params2, reductor_params)[0]
            state_pure = rm.circuit1()
            state_pure = rm.circuit2(state_pure)

            KLs.append(KL_divergence(np.abs(state_cut)**2, np.abs(state_pure)**2))



'''ax1.text(0.5, max_entries / 2 ** qubits * 1.005, r'Max $CB_\epsilon$ rank allowed', size=12)
ax1.text(0.1, confident_entries / 2**qubits * 1.005 , r'Max $CB_\epsilon$ rank estimable', size=12)'''

#ax1.text(0.6*lenM, confident_entries / 2**qubits * 1.005 , 'Counter %s'%counter)

ax1.text(.4*lenM,confident_entries / 2**qubits * .25, r'$KL = $' + '%.4f'%(np.mean(KLs)) + r'$\pm$' + '%.4f'%(np.std(KLs)), 
bbox=dict(boxstyle="round",
                   fc=(.9, 0.9, 0.9),
                   ec=(.1, 0.1, 0.1),
                   ), size=12)

fig.savefig(folder + '/%sQ_%sr_%scb_%seps.pdf'%(qubits, random_layers, cb_layers, epsilon), bbox_inches='tight')


legfig = plt.figure()
print(leg)
legfig.legend(*leg, ncol=2)
legfig.savefig(folder + '/legend.pdf', bbox_inches='tight')

'''fig.savefig('results/random_init/figures/%sQ_%sr_%seps.pdf'%(qubits, random_layers, epsilon))

ax4.hist(KLs[:, -1], bins=np.logspace(-1.75,-0.9, 15), color='black', rwidth=0.95)
ax4.set_xscale('log')
ax4.set_xlabel('KL divergence')
ax4.set_ylabel('N appearences')
figkl.savefig('results/random_init/figures/%sQ_%sr_%seps_kl.pdf'%(qubits, random_layers, epsilon))'''
