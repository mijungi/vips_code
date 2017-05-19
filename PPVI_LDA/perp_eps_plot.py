# to plot perplexity as a function of epsilon
# Mijung wrote on the 26th of Aug, 2016

import numpy as np
import matplotlib.pyplot as plt

D = 200000

seednum = 0

batchmat = [10, 20, 50, 100, 200, 400]
Jmat = [0.2]
epsilonmat = [0.2, 0.4, 0.6, 0.8, 1]

# priv = 0
# epsilon = 0
# comp = 0

priv = 1
# epsilon = 2
comp = 2

for S in batchmat:

    batchsize = S
    documentstoanalyze = int(Jmat[0]*D/float(batchsize))

    # if S==50:
    #     plt.subplot(221)
    # elif S==100:
    #     plt.subplot(222)
    # elif S==200:
    #     plt.subplot(223)
    # else:
    #     plt.subplot(224)

    for eps in epsilonmat:
        epsilon = eps
        method = 'static_results/static_private_seed=%s_J=%s_S=%s_priv=%s_epsilon=%s_compo=%s_D=%s' %(seednum, documentstoanalyze, batchsize, priv, epsilon, comp, D)
        pp = np.load(method+'.npy')
        # mean_perp = np.mean(pp[-2:])

        if S==10:
            mean_perp = np.mean(pp[-32:])
            l00, = plt.plot(eps, mean_perp, 'ro--',  label = 'S_%s' %(batchsize))
        elif S==20:
            mean_perp = np.mean(pp[-32:])
            l0, = plt.plot(eps, mean_perp, 'ro--',  label = 'S_%s' %(batchsize))
        elif S==50:
            mean_perp = np.mean(pp[-32:])
            l, = plt.plot(eps, mean_perp, 'ro',  label = 'S_%s' %(batchsize))
        elif S==100:
            mean_perp = np.mean(pp[-16:])
            ll, = plt.plot(eps, mean_perp, 'go',  label = 'S_%s' %(batchsize))
        elif S==200:
            mean_perp = np.mean(pp[-8:])
            lll, = plt.plot(eps, mean_perp, 'bo',  label = 'S_%s' %(batchsize))
        else:
            mean_perp = np.mean(pp[-4:])
            llll, = plt.plot(eps, mean_perp, 'ko',  label = 'S_%s' %(batchsize))

    # method = 'static_results/static_nonprivate_seed=%s_J=%s_S=%s_priv=%s_epsilon=%s_compo=%s_D=%s' %(seednum, documentstoanalyze, batchsize, 0, 0, 0, D)
    # pp = np.load(method+'.npy')
    # # mean_perp = np.mean(pp[-10])
    # len_eps = len(epsilonmat)
    # if S==50:
    #     mean_perp = np.mean(pp[-16:])
    #     npv1, = plt.plot(epsilonmat, mean_perp*np.ones(len_eps), label = 'nonprivate_S=%s' %(batchsize))
    # elif S==100:
    #     mean_perp = np.mean(pp[-8:])
    #     npv2, = plt.plot(epsilonmat, mean_perp*np.ones(len_eps), label = 'nonprivate_S=%s' %(batchsize))
    # elif S==200:
    #     mean_perp = np.mean(pp[-4:])
    #     npv3, = plt.plot(epsilonmat, mean_perp*np.ones(len_eps), label = 'nonprivate_S=%s' %(batchsize))
    # else:
    #     mean_perp = np.mean(pp[-2:])
    #     npv4, = plt.plot(epsilonmat, mean_perp*np.ones(len_eps), label = 'nonprivate_S=%s' %(batchsize))

plt.grid()

plt.title('total Doc=200K, #doc looked at =%s' %(Jmat[0]*D))
plt.xlabel('epsilon')
plt.ylabel('perplexity')
plt.legend(handles=[l00, l0, l,ll, lll, llll])
# plt.legend(handles=[l,ll, lll, llll, npv1, npv2, npv3, npv4])
plt.xlim([0.1, 1.1])
plt.ylim([1100, 2000])
# plt.yscale('log')
plt.show()