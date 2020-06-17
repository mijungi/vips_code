# __author__ = 'mijung'

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

import cal_amp_eps
import calculate_moments_accountant as MA
import calculate_privacy_burned as cal_pri

# epsilon = 0.5 # total privacy budget
T = 16000

total_del = 1e-4  # per-iteration budget
q = 10 / 400000.0

sigma = 1 + 1e-10

del_iter = 1e-6
total_eps_MA = 1.0
res = minimize_scalar(cal_pri.strong_composition, bounds=(0, 10),
                      args=(total_eps_MA, total_del, T, q, del_iter),
                      method='bounded')
eps_iter = res.x

print(eps_iter)
