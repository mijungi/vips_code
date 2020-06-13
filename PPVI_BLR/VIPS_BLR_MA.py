# import matplotlib.pyplot as plt
# plt.ion()

import pickle as pickle

import numpy as np
import numpy.random as rn
import scipy.linalg as sl

mvnrnd = rn.multivariate_normal

from generateData import *

############################################
""" define a few functions """


############################################

def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def vec(x):
	return np.reshape(x, (np.prod(x.shape)))


def max_norm1_pre_process(X):
	X = X.toarray()
	normaliser = np.max(np.sqrt(np.sum(np.abs(X) ** 2, axis=-1)))
	X = X / normaliser
	return X


def computeOdds(X, theta):
	return sigmoid(np.dot(X, theta))


def mean_squared_error(ytst, ypred, Ntst):
	return np.sqrt(np.sum((ytst - ypred) ** 2) / np.float(Ntst))


def Laplace_noise(privacy_bugdet, L1_sensitivity, dim):
	return rn.laplace(0, L1_sensitivity / np.float(privacy_bugdet), dim)


############################################
""" variational Bayes EM code """


############################################


def VBEstep_private(sigma, X, y, exp_nat_params, totN):
	# private version
	# exp_nat_params: <theta theta\trp>_q(theta)

	N = np.size(X, 0)
	d = np.size(X, 1)

	""" compute s_1 and perturb it """
	Xy = np.dot(X.T, y)
	Xsum = np.sum(X, 0)
	exp_suff_stats1 = (Xy - 0.5 * Xsum) / np.float(N)  # first moment

	sensitivity = 1 / np.float(totN)
	nsv = (sensitivity ** 2) * (sigma ** 2)
	noise = np.random.normal(0, nsv, exp_suff_stats1.shape)

	exp_suff_stats1_tile = noise + exp_suff_stats1

	""" compute s_2 and perturb it """
	X1 = np.dot(np.dot(X, exp_nat_params), X.T)
	arg_i = np.sqrt(np.diag(X1))
	mean_xi_i = 0.5 * np.tanh(0.5 * arg_i) / arg_i
	ind_zero = np.nonzero(arg_i == 0)
	mean_xi_i[ind_zero] = 0.25

	exp_suff_stats2 = np.dot(X.T * mean_xi_i, X)
	exp_suff_stats2 = exp_suff_stats2 / np.float(N)

	nsv = ((sensitivity / 2) ** 2) * (sigma ** 2)
	nse_mat = np.random.normal(0, nsv, exp_suff_stats2.shape)
	upper_nse_mat = np.triu(nse_mat, 0)

	for i in range(d):
		for j in range(i, d):
			upper_nse_mat[j][i] = upper_nse_mat[i][j]

	nse = upper_nse_mat

	exp_suff_stats2_tile = exp_suff_stats2 + nse

	if sigma > 0:
		# to ensure the matrix is positive definite
		w, v = np.linalg.eig(exp_suff_stats2_tile)
		# remember: XX_tile = np.dot(v, np.dot(np.diag(w), v.transpose()))
		neg_idx = np.nonzero(w < 0)
		w[neg_idx] = 0.0001

		exp_suff_stats2_tile = np.dot(v, np.dot(np.diag(w), v.transpose()))

	return exp_suff_stats1_tile, exp_suff_stats2_tile


def VBMstep_stochastic(rhot, nu_old, ab_old, N, a0, b0, exp_suff_stats1, exp_suff_stats2, mean_alpha_prv, iter):
	# rhot is the decreasing step-size

	d = np.size(exp_suff_stats1)

	nu1 = N * exp_suff_stats1
	nu2 = mean_alpha_prv * np.eye(d) + N * exp_suff_stats2

	nu = np.concatenate((nu1, vec(nu2)))
	if iter == 0:
		nu_new = nu
	else:
		nu_new = nu_old * (1 - rhot) + rhot * nu

	nu1_new = nu_new[0:d]
	nu2_new = np.reshape(nu_new[d:np.size(nu_new)], (d, d))

	Cov_theta_inv = nu2_new
	# Cov_theta_inv = mean_alpha_prv*np.eye(d) + N*exp_suff_stats2
	Cov_theta = np.linalg.inv(Cov_theta_inv)

	Mu_theta = np.dot(Cov_theta, nu1_new)
	# Mu_theta = np.dot(Cov_theta, N*exp_suff_stats1)

	exp_nat_params = Cov_theta + np.outer(Mu_theta, Mu_theta)

	# update aN and bN
	if iter == 0:

		aN = a0 + 0.5 * d
		bN = b0 + 0.5 * (np.dot(Mu_theta, Mu_theta) + np.trace(Cov_theta))
		mean_alpha = aN / np.float(bN)
		ab_new = np.array([aN, bN])

	else:

		aN = a0 + 0.5 * d
		bN = b0 + 0.5 * (np.dot(Mu_theta, Mu_theta) + np.trace(Cov_theta))
		ab = np.array([aN, bN])
		# ab_new = ab
		ab_new = ab_old * (1 - rhot) + ab * rhot
		mean_alpha = ab_new[0] / np.float(ab_new[1])

	return nu_new, ab_new, exp_nat_params, mean_alpha, Mu_theta
