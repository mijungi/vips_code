# to show how per-iteration budget looks in terms of S/D and J.
# Mijung wrote on the 26th of Aug, 2016

import numpy
import matplotlib.pyplot as plt

N = 10000
D = 200000

Smat = [10, 20, 50, 100, 200, 400]
Jmat = [0.1, 0.2, 0.4]
# Jmat = [0.2]
# Jmat = [0.4]
# Jmat = [0.8]

# Jmat = [200, 400, 800]

sen = N/float(D)
epsilon = 1
delta = 0.000001


for Jlist in Jmat:
    for S in Smat:
        nu = S/float(D)
        J = int(Jlist*D/float(S))
        budget_cdp = numpy.sqrt(2*epsilon)/float(2*nu*numpy.sqrt(J))
        budget_adv = epsilon/float(4*nu*numpy.sqrt(2*J*numpy.log(1/delta)))
        budget_linear = epsilon/float(2*J*nu)
        print(budget_linear)

        if S==10:
            cdp00, = plt.plot(Jlist, budget_cdp, 'o', color = (1, 0.6, 0), label = 'cdp_S_%s' %(S))
            adv00, = plt.plot(Jlist, budget_adv, 'o', color = (0, 0.6, 1), label = 'adv_S_%s' %(S))
        elif S==20:
            cdp0, = plt.plot(Jlist, budget_cdp, 'o', color = (1, 1, 0), label = 'cdp_S_%s' %(S))
            adv0, = plt.plot(Jlist, budget_adv, 'o', color = (0, 1, 1), label = 'adv_S_%s' %(S))
        elif S==50:
            cdp1, = plt.plot(Jlist, budget_cdp, 'o', color = (1, 0, 0), label = 'cdp_S_%s' %(S))
            adv1, = plt.plot(Jlist, budget_adv, 'o', color = (0, 0.35, 0), label = 'adv_S_%s' %(S))
            lin1, = plt.plot(Jlist, budget_linear, 'o', color = (0, 0, 1), label = 'lin')
        elif S==100:
            cdp2, = plt.plot(Jlist, budget_cdp, 'o', color = (0.6, 0, 0),  label = 'cdp_S_%s' %(S))
            adv2, = plt.plot(Jlist, budget_adv, 'o', color = (0, 0.55, 0), label = 'adv_S_%s' %(S))
            # lin2, = plt.plot(Jlist, budget_linear, 'o', color = (0, 0, 0.7), label = 'lin_S_%s' %(S))
        elif S==200:
            cdp3, = plt.plot(Jlist, budget_cdp, 'o', color = (0.4, 0, 0),  label = 'cdp_S_%s' %(S))
            adv3, = plt.plot(Jlist, budget_adv, 'o', color = (0, 0.75, 0), label = 'adv_S_%s' %(S))
            # lin3, = plt.plot(Jlist, budget_linear, 'o', color = (0, 0, 0.4), label = 'lin_S_%s' %(S))
        else:
            cdp4, = plt.plot(Jlist, budget_cdp, 'o', color = (0.2, 0, 0),  label = 'cdp_S_%s' %(S))
            adv4, = plt.plot(Jlist, budget_adv, 'o', color = (0, 1, 0), label = 'adv_S_%s' %(S))
            # lin4, = plt.plot(Jlist, budget_linear, 'o', color = (0, 0, 0.2), label = 'lin_S_%s' %(S))
            # plt.plot(Jlist, budget_adv, 'g')
            # plt.plot(Jlist, budget_linear, 'b')

#         if S==50:
#             plt.subplot(221)
#             # plt.title('#doc looked at: %s' %(int(Jlist*D)))
#             plt.ylabel('per-iteration-budget')
#             # plt.xticks(Smat, ('50', '100', '200', '400', '800'))
#             # plt.xticks(Jmat, ('0.1', '0.2', '0.4'))
#             # plt.title('percentage of doc looked at (%s x D)' %(Jlist))
#         elif S==100:
#             # plt.title('#doc looked at: %s' %(int(Jlist*D)))
#             plt.subplot(222)
#             # plt.xticks(Smat, ('50', '100', '200', '400', '800'))
#             # plt.xticks(Jmat, ('0.1', '0.2', '0.4'))
#             # plt.title('percentage of doc looked at (%s x D)' %(Jlist))
#         elif S==200:
#             plt.subplot(223)
#             # plt.title('#doc looked at: %s' %(int(Jlist*D)))
#             # plt.xlabel('batchsize')
#             plt.ylabel('per-iteration-budget')
#             # plt.xticks(Smat, ('50', '100', '200', '400', '800'))
#             # plt.xticks(Jmat, ('0.1', '0.2', '0.4'))
#             # plt.title('percentage of doc looked at (%s x D)' %(Jlist))
#         else:
#             plt.subplot(224)
            # plt.title('#doc looked at: %s' %(int(Jlist*D)))
            # plt.xlabel('batchsize')
            # plt.xticks(Smat, ('50', '100', '200', '400', '800'))
            # plt.xticks(Jmat, ('0.1', '0.2', '0.4'))
            # plt.title('percentage of doc looked at (%s x D)' %(Jlist))
        # print [S, J, budget_cdp, budget_adv, budget_linear]

        # plt.title('# iter: %s' %(J))
        # plt.title('percentage of doc looked at (%s x D)' %(Jlist))
        plt.xticks(Jmat, ('0.1', '0.2', '0.4'))
        # plt.plot(Jlist, budget_cdp, 'ro-')
        # plt.plot(Jlist, budget_adv, 'go')
        # plt.plot(Jlist, budget_linear, 'bo')
        # plt.xscale('log')
        plt.yscale('log')
        plt.grid()
        plt.ylim([0.5,400])
        plt.xlim([0, 0.7])
    plt.ylabel('per-iteration-budget')
    plt.legend(handles=[cdp00, cdp0, cdp1, cdp2, cdp3, cdp4, adv00, adv0, adv1, adv2, adv3, adv4, lin1]) # lin2, lin3, lin4
    plt.xlabel('proportion of #doc seen')

plt.title('D=200,000, max-length of words in each doc:10,000')
plt.savefig('budget_plot.pdf')
# plt.show()