# __author__ = 'mijung'
# for testing VIPS for the stochastic setting under Bayesian Logistic Regression
# Oct 27, 2016

import os
import sys

import VIPS_BLR  # this has all core functions
import generateData
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rn
import scipy
import scipy.io
from scipy.optimize import minimize_scalar
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc

import cal_amp_eps

mvnrnd = rn.multivariate_normal

if __name__ == '__main__':

	""" inputs """
	# rn.seed(10)
	rnd_num = np.random.randint(1000000)
	rn.seed(rnd_num)

	""" data generation """
	N = 200
	Ntst = 50
	d = 20

	""" hyper-params for the prior over the parameters """
	alpha = 0.02
	a0 = 1.
	b0 = 1.

	y, X, true_theta = generateData.generate_data(d, N, alpha)
	ytst, Xtst = generateData.generate_testdata(Ntst, true_theta)

	""" stochastic version """
	tau0 = 1024
	kappa = 0.7
	MaxIter = 40  # EM iteration
	S = 20
	nu = S / float(N)

	""" (1) we test the non-private version ! """
	print('we first test a non-private version!')
	eps_iter = 0
	delta_iter = 0

	exp_nat_params_prv = np.ones([d, d])
	mean_alpha_prv = a0 / b0
	auc_nonprivate_stoch = np.zeros(MaxIter)

	for iter in range(MaxIter):
		# print(iter, 'th iteration')
		# iterations start here

		rhot = (tau0 + iter) ** (-kappa)

		""" select a new mini-batch """
		rand_perm_nums = np.random.permutation(N)
		idx_minibatch = rand_perm_nums[0:S]
		xtrain_m = X[idx_minibatch, :]
		ytrain_m = y[idx_minibatch]

		exp_suff_stats1, exp_suff_stats2 = VIPS_BLR.VBEstep_private(eps_iter, delta_iter, xtrain_m, ytrain_m,
		                                                            exp_nat_params_prv, N)

		if iter == 0:
			nu_old = []
			ab_old = []
		nu_new, ab_new, exp_nat_params, mean_alpha, Mu_theta = VIPS_BLR.VBMstep_stochastic(rhot, nu_old, ab_old, N, a0,
		                                                                                   b0, exp_suff_stats1,
		                                                                                   exp_suff_stats2,
		                                                                                   mean_alpha_prv, iter)

		mean_alpha_prv = mean_alpha
		exp_nat_params_prv = exp_nat_params
		nu_old = nu_new
		ab_old = ab_new

		""" compute roc_curve and auc """
		ypred = VIPS_BLR.computeOdds(Xtst, Mu_theta)
		fal_pos_rate_tst, true_pos_rate_tst, thrsld_tst = roc_curve(ytst, ypred.flatten())
		auc_tst = auc(fal_pos_rate_tst, true_pos_rate_tst, reorder=True)
		auc_nonprivate_stoch[iter] = auc_tst

		# update iteration number
		iter = iter + 1

	print(('AUC under the non-private version is', auc_nonprivate_stoch[-1]))

	""" (2) we test the private version ! """
	print('now we test a private version!')

	# calculate the per-iteration privacy budget
	comp = 2  # cdp
	epsilon = 0.1

	res = minimize_scalar(cal_amp_eps.f, bounds=(0, 400), args=(epsilon, comp, MaxIter, nu), method='bounded')
	eps_iter = res.x
	print(('Assigned per-iteration budget is ', eps_iter))
	delta = 0.000001
	if comp == 0:
		delta_iter = delta / (MaxIter * nu)
	else:
		delta_iter = 0.0001

	# now we start the iteration

	exp_nat_params_prv = np.ones([d, d])
	mean_alpha_prv = a0 / b0
	auc_private_stoch_zcdp = np.zeros(MaxIter)

	for iter in range(MaxIter):

		# iterations start here

		rhot = (tau0 + iter) ** (-kappa)

		""" select a new mini-batch """
		rand_perm_nums = np.random.permutation(N)
		idx_minibatch = rand_perm_nums[0:S]
		xtrain_m = X[idx_minibatch, :]
		ytrain_m = y[idx_minibatch]

		exp_suff_stats1, exp_suff_stats2 = VIPS_BLR.VBEstep_private(eps_iter, delta_iter, xtrain_m, ytrain_m,
		                                                            exp_nat_params_prv, N)

		if iter == 0:
			nu_old = []
			ab_old = []
		nu_new, ab_new, exp_nat_params, mean_alpha, Mu_theta = VIPS_BLR.VBMstep_stochastic(rhot, nu_old, ab_old, N, a0,
		                                                                                   b0, exp_suff_stats1,
		                                                                                   exp_suff_stats2,
		                                                                                   mean_alpha_prv, iter)

		mean_alpha_prv = mean_alpha
		exp_nat_params_prv = exp_nat_params
		nu_old = nu_new
		ab_old = ab_new

		""" compute roc_curve and auc """
		ypred = VIPS_BLR.computeOdds(Xtst, Mu_theta)
		fal_pos_rate_tst, true_pos_rate_tst, thrsld_tst = roc_curve(ytst, ypred.flatten())
		auc_tst = auc(fal_pos_rate_tst, true_pos_rate_tst, reorder=True)
		auc_private_stoch_zcdp[iter] = auc_tst

		# update iteration number
		iter = iter + 1

	print(('AUC under the private version is', auc_private_stoch_zcdp[-1]))
