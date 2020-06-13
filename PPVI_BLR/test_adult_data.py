# __author__ = 'mijung'
# for testing VIPS for adult data
# written on July 2, 2019

import VIPS_BLR_MA # this has all core functions
import os
import sys
import scipy
import scipy.io
import numpy as np
import numpy.random as rn
from sklearn.metrics import roc_curve,auc
from sklearn import preprocessing
import cal_amp_eps
from scipy.optimize import minimize_scalar
import generateData
import matplotlib.pyplot as plt
import pickle

mvnrnd = rn.multivariate_normal

if  __name__ =='__main__':

    """ inputs """
    # rn.seed(10)
    # rnd_num = np.random.randint(1000000)
    rnd_num = 123
    rn.seed(rnd_num)

    """ load data """

    filename = 'adult.p'

    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()

    # unpack data
    y_tot, x_tot = data
    N_tot, d = x_tot.shape

    N = int(0.8 * N_tot)
    N_test = N_tot - N

    X = x_tot[:N, :]
    y = y_tot[:N]
    Xtst = x_tot[N:, :]
    ytst = y_tot[N:]

    """ hyper-params for the prior over the parameters """
    alpha = 0.02
    a0 = 1.
    b0 = 1.

    """ stochastic version """
    tau0 = 1024
    kappa = 0.7
    MaxIter = 100 # EM iteration
    # S = 20
    nu = 0.005
    S =  np.int(nu*N)
    print(S)

    """ we test the private version ! """
    # sigma = 10 # privacy parameter

    # now we start the iteration

    exp_nat_params_prv = np.ones([d,d])
    mean_alpha_prv = a0/b0

    num_repeat = 20

    iter_sigmas = np.array([0., 1., 10., 50.])
    auc_private_stoch_ours = np.empty([iter_sigmas.shape[0], num_repeat])

    for k in range(iter_sigmas.shape[0]):
        sigma = iter_sigmas[k]

        for repeat_idx in range(num_repeat):


            for iter in range(MaxIter):

                # iterations start here

                rhot = (tau0+iter)**(-kappa)

                """ select a new mini-batch """
                rand_perm_nums =  np.random.permutation(N)
                idx_minibatch = rand_perm_nums[0:S]
                xtrain_m = X[idx_minibatch,:]
                ytrain_m = y[idx_minibatch]

                exp_suff_stats1, exp_suff_stats2 = VIPS_BLR_MA.VBEstep_private(sigma, xtrain_m, ytrain_m, exp_nat_params_prv, S)

                if iter==0:
                    nu_old = []
                    ab_old = []
                nu_new, ab_new, exp_nat_params, mean_alpha, Mu_theta = VIPS_BLR_MA.VBMstep_stochastic(rhot, nu_old, ab_old, N, a0, b0, exp_suff_stats1, exp_suff_stats2, mean_alpha_prv, iter)

                mean_alpha_prv = mean_alpha
                exp_nat_params_prv = exp_nat_params
                nu_old = nu_new
                ab_old = ab_new

                """ compute roc_curve and auc """
                ypred = VIPS_BLR_MA.computeOdds(Xtst, Mu_theta)
                # ypred = 1 * (ypred >= 0.5)
                fal_pos_rate_tst, true_pos_rate_tst, thrsld_tst = roc_curve(ytst, ypred.flatten())
                auc_tst = auc(fal_pos_rate_tst,true_pos_rate_tst,reorder=True)

                # ypred = 1*(ypred>=0.5)
                # acc = np.sum(ytst == ypred) / np.float(N_test)
                # auc_private_stoch_ours[iter]=acc

                # update iteration number
                iter = iter + 1

            print(('AUC is', auc_tst))
            print(('sigma is', sigma))

            auc_private_stoch_ours[k, repeat_idx] = auc_tst

    np.save('accuracy_ours', auc_private_stoch_ours)