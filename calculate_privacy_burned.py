import numpy as np

import calculate_moments_accountant as MA

def strong_composition(x, total_eps, total_del, J, nu, del_iter):

    # x is per-iteration budget

    # (1) privacy amplification due to sub-sampling
    eps_amp = np.log(1 + nu*(np.exp(x)-1))
    del_amp = nu * del_iter

    del_new = total_del - J * del_amp
    if del_new < 0:
        print "del_new is negative"
        print "choose a different del_iter"

    # (2) Strong composition
    eps_0 = np.exp(eps_amp) - 1
    eps_strong_comp = np.sqrt(2 * J * np.log(1 / del_new)) * eps_amp + J * eps_amp * eps_0

    # print eps_strong_comp
    # print total_eps

    return (eps_strong_comp-total_eps)**2

def moments_accountant(sigma, total_del, nu, J):

    # sigma has to be greater than equal to 1
    # also has to be smaller than 1/(16*q)
    # sigma = 3 + 1e-6 # make sure sigma < 1/(16*q) and sigma>=1
    if sigma > 1 / (16 * nu):
        print "choose a smaller sigma"

    # make sure lambda < sigma^2 log (1/(q*sigma))
    max_lmbd = int(np.floor((sigma ** 2) * np.log(1 / (nu * sigma))))
    print max_lmbd

    lmbds = xrange(1, max_lmbd + 1)
    log_moments = []
    for lmbd in lmbds:
        log_moment = 0
        log_moment = MA.compute_log_moment(nu, sigma, J, lmbd, verify=True, verbose=True)
        log_moments.append((lmbd, log_moment))

    total_epsilon, total_delta = MA.get_privacy_spent(log_moments, target_delta=total_del)

    return total_epsilon