# __author__ = 'mijung'
# to calculate amplified epsilon according to which composition theorem to use
# wrote on the 31st of Aug, 2016

import matplotlib.pyplot as plt
import numpy as np
# import theano
# import theano.tensor as T
import scipy
from scipy.optimize import minimize_scalar

""" this is for (epsilon, delta)-DP mechanisms """


def f(x, total_eps, which_comp, J, nu):
	# calculate eps_amp first
	# trm1 = 1 + nu*(np.exp(x)-1)
	# trm2 = 1 - nu*(1-np.exp(-x))
	# # print trm1, trm2
	# eps_amp = np.log(trm1) - np.log(trm2)
	eps_amp = np.log(1 + nu * (np.exp(x) - 1))

	# (1) linear composition
	if which_comp == 0:
		# linear composition
		return (total_eps - J * eps_amp) ** 2

	elif which_comp == 1:
		# advanced composition
		delta_iter = 0.000001
		delta = 0.0001 - J * nu * delta_iter
		if delta < 0:
			print("delta is less than 0")

		return (total_eps - np.sqrt(2 * J * np.log(1 / delta)) * eps_amp - J * eps_amp * (np.exp(eps_amp) - 1)) ** 2

	else:
		# CDP composition
		delta_iter = 0.000001
		delta_amp = nu * delta_iter
		c2 = 2 * np.log(1.25 / delta_amp)
		delta = 0.0001
		rho = J * (eps_amp ** 2) / (2 * c2)

		return (total_eps - (rho + 2 * np.sqrt(rho * np.log(1 / delta)))) ** 2
