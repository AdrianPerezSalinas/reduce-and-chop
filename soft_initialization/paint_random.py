qubits = 8
random_layers = 5
seeds = [1, 51]
cb_layers = 2
ansatz = 'tfim'
import matplotlib.pyplot as plt
import numpy as np

fig, ax1 = plt.subplots(ncols=1, figsize=(4.5, 3))
#figkl, ax3= plt.subplots()

folder = "results_soft_init/random_tfim"
folder_ = "results_soft_init/random_tfim"

'''folder = "results_soft_init/random_red/" + ansatz
folder_ = "results_soft_init/random_red/" + ansatz
'''
k = -1

for epsilon in [1,2, 3, 5, 10, 15]:
    k += 1
    Cb_ranks = []
    Costs = []
    kl = []
    for s in range(seeds[0], seeds[1]):
        cb_ranks = []
        costs = []
        changes = [0]
        with open(folder + "/%sQ_%sr_%sCB_%seps_%s.txt"%(qubits, random_layers, cb_layers, epsilon, s), "r") as f:
            for i, line in enumerate(f.readlines()):
                l = line.split('\n')[0]
                l = line.split(' ')
                cb_ranks.append(float(l[0]) / 2 ** qubits)
                #costs.append(max(float(l[1]), float(l[0]) - 6))
                costs.append(float(l[1]) / 2 ** qubits)


        Costs.append(costs)
        Cb_ranks.append(cb_ranks)

        #with open(folder + "/%sQ_%sr_%sCB_%s.0eps_%s.json"%(qubits, random_layers, cb_layers, epsilon, s), "rb") as f:
        #    results = pickle.load(f)

        #rm = cut_model(qubits, random_layers, cb_layers, epsilon=epsilon)

        #rm.compute_target(rm.joint_cut(results['params1'], results['params2']))
        #prob = rm.execute(results['params1'], results['params2'], results['x'])[0]
        #_ = rm.KL_divergence(prob, rm.target_distr)
        #kl.append(_)

    lens = [len(costs) for costs in Costs]
    lenM = max(lens)
    for i in range(len(Costs)):
        Costs[i] += [Costs[i][-1]] * (lenM - len(Costs[i]))
        Cb_ranks[i] += [Cb_ranks[i][-1]] * (lenM - len(Cb_ranks[i]))


    Cb_ranks = np.array(Cb_ranks)
    Costs = np.array(Costs)
    iters = np.arange(len(Cb_ranks[0]))

    Cb_ranks = np.sort(Cb_ranks, axis=0)
    Costs = np.sort(Costs, axis=0)

    ax1.plot(iters, np.mean(Cb_ranks, axis=0), color='C%s'%(k), alpha=1, label = r'$\epsilon =$'+ str(epsilon) + ' %')
    ax1.fill_between(iters,Cb_ranks[0], Cb_ranks[-1], color='C%s'%(k), alpha = 0.25)

#plt.fill_between(iters,np.mean(Cb_ranks, axis=0), np.mean(Cb_ranks, axis=0) - np.var(Cb_ranks, axis=0), alpha = 0.25)
    #plt.plot(Costs[i], label='Costs', color='black', alpha=0.25)

#ax3.hist(kl, bins=np.logspace(-1, 0, 25), color='C%s'%(cb_layers-1), alpha=0.5, label = 'Depth = %s'%cb_layers)

pos = ax1.get_position()
pos.x0 += 0.05
pos.x1 += 0.05
pos.y0 += 0.075
pos.y1 += 0.05
ax1.set_position(pos)


ax1.set_xlabel('Optimizer iterations', fontsize=15)
ax1.set_ylabel(r'$CB_\varepsilon / 2^n$', fontsize=15)
ax1.set_ylim([0, 1])
ax1.set_xlim([-1, len(Cb_ranks[0])])

ax1.tick_params(axis='x', labelsize = 14)
ax1.tick_params(axis='y', labelsize = 14)

from matplotlib.lines import Line2D
custom_lines = []
labels = []
for i, eps in enumerate([1,2,3,5,10,15]):
    custom_lines.append(Line2D([0], [0], color='C%s'%i, lw=2))
    labels.append(r'$\epsilon =$'+ str(eps) + ' %') 

# ax1.legend(custom_lines, labels, ncol=2, handlelength = 1)

#ax2.legend()
    

#ax1.set_yscale('log')
#ax3.legend()
#ax3.set_xscale('log')

#fig.suptitle(r'Costs and CB-ranks for a random circuit, ' + string)

# add custom legendplt.legend()
#plt.yscale('log')

fig.savefig(folder_ + "/%sQ_%sr_%sCB_"%(qubits, random_layers, cb_layers) + ansatz + ".pdf")

'''fig.savefig('results/random/figures/%sQ_%sr_%seps.pdf'%(qubits, random_layers, epsilon))
figkl.savefig('results/random/figures/%sQ_%sr_%seps_kl.pdf'%(qubits, random_layers, epsilon))'''
