# __author__ = 'mijung'
# to calculate amplified epsilon according to which composition theorem to use
# wrote on the 31st of Aug, 2016

import numpy as np
import matplotlib.pyplot as plt
#import theano
#import theano.tensor as T
import scipy
from scipy.optimize import minimize_scalar


# de = T.scalar("de")
# iter = T.scalar("iter")
# e = T.scalar("e")
# eps_amp = theano.shared(np.asarray(0.), name='eps_amp')
#
# # error = T.mean((e - T.sqrt(2*j*T.log(1/de))*eps_amp - j*eps_amp*(T.exp(eps_amp)-1))**2)
# error = e - iter*eps_amp + 0*de
# lr = 0.001
# grad = T.grad(cost=error, wrt=eps_amp)
# f = theano.function(inputs=[e, iter, de], outputs = error, updates=[[eps_amp, eps_amp - grad*lr]])
# # f = theano.function(inputs=[e, j, de], outputs = error)
#
#
# iter_steps = 100
# for i in range(iter_steps):
#     # print i
#     err = f(total_eps, J, delta)
#     print i, err
#
# print eps_amp.get_value()

# def f_esp_prime(x, tot_esp, nu):
#
#     # calculate eps_amp first
#     trm1 = 1 + nu*(np.exp(x)-1)
#     trm2 = 1 - nu*(1-np.exp(-x))
#     # print trm1, trm2
#     return (tot_esp - np.log(trm1/trm2))**2
#
#     # return eps_amp



""" this is for (epsilon, delta)-DP mechanisms """
def f(x, total_eps, which_comp, J, nu):

    # calculate eps_amp first
    # trm1 = 1 + nu*(np.exp(x)-1)
    # trm2 = 1 - nu*(1-np.exp(-x))
    # # print trm1, trm2
    # eps_amp = np.log(trm1) - np.log(trm2)
    eps_amp = np.log(1 + nu*(np.exp(x)-1))

    #(1) linear composition
    if which_comp==0:
        # linear composition
            return (total_eps - J*eps_amp)**2

    elif which_comp==1:
        # advanced composition
        delta_iter = 0.000001
        delta = 0.0001 - J*nu*delta_iter
        if delta<0:
            print "delta is less than 0"

        return (total_eps - np.sqrt(2*J*np.log(1/delta))*eps_amp - J*eps_amp*(np.exp(eps_amp)-1))**2

    else:
        # CDP composition
        delta_iter = 0.000001
        delta_amp = nu*delta_iter
        c2 = 2*np.log(1.25/delta_amp)
        delta = 0.0001
        rho = J*(eps_amp**2)/(2*c2)

        return (total_eps - (rho + 2*np.sqrt(rho*np.log(1/delta))))**2


def f_1(x, total_eps, which_comp, J, nu):

    # calculate eps_amp first
    trm1 = 1 + nu*(np.exp(x)-1)
    trm2 = 1 - nu*(1-np.exp(-x))
    # print trm1, trm2
    eps_amp = np.log(trm1) - np.log(trm2)
    # =print eps_amp

    #(1) linear composition
    if which_comp==0:
        # linear composition
            return (total_eps - J*eps_amp)**2

    elif which_comp==1:
        # advanced composition
        delta = 0.000001
        return (total_eps - np.sqrt(2*J*np.log(1/delta))*eps_amp - J*eps_amp*(np.exp(eps_amp)-1))**2

    else:
        # CDP composition
        # return (total_eps - 0.5*J*eps_amp**2)**2
        return (total_eps - 0.5*J*eps_amp*(np.exp(eps_amp)-1))**2





# total_eps = 1
# # which_comp = 2
#
# D = 400000
# S = 10
# # J = int(0.01*(D)/float(S))
# nu = S/float(D) # sampling ratio
#
# res = minimize_scalar(f_esp_prime, bounds=(0, 400), args=(total_eps, nu), method='bounded')
# print res, res.x

# nu_mat = [0.000001, 0.0001, 0.01, 0.1, 0.2, 0.4, 0.8, 1]
# eps_amp_mat = np.zeros(len(nu_mat))
#
# j = 0
# for i in nu_mat:
#     eps_amp_mat[i] = f_esp_prime(1, nu_mat[j])
#     j = j + 1
#
# print eps_amp_mat
# plt.plot(nu_mat, eps_amp_mat)
# plt.xscale('log')
# plt.show()
