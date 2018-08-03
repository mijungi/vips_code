

from scipy.optimize import minimize_scalar
import calculate_privacy_burned as cal_pri
import numpy as np
import scipy.io

D = 60000
Hmat = [50, 100]
Smat = [50, 100, 200]
Itermat = 0.25*D

ndims = 784

# totEps = 0.5
totDel = 1e-4

ntrain = 50000

# sigma = 2
sigma = 1 + 1e-6
del_iter = 1e-9


for K in Hmat:
    for S in Smat:

        maxit = int(Itermat / float(S))
        numIter = int(maxit * (K + ndims + 5))
        print numIter

        sampRate = S / float(ntrain)

        """ privacy budget calculation """
        # (1) to set the same level of burned privacy, we first calculate MA composition
        total_eps_MA = cal_pri.moments_accountant(sigma, totDel, sampRate, numIter)
        c2 = 2 * np.log(1.25 / del_iter)
        eps_iter = np.sqrt(c2) / sigma
        budget_MA = [eps_iter, del_iter, total_eps_MA]
        print budget_MA

        """save results"""
        method = 'privacy_budget_MA_S=%s_K=%s_sigma=%s' % (S,K,int(sigma))
        # np.save(method + '.npy', budget_MA)
        scipy.io.savemat(method, dict(budget_MA=budget_MA))

        # (2) strong composition
        res = minimize_scalar(cal_pri.strong_composition, bounds=(0, 50), args=(total_eps_MA, totDel, numIter, sampRate, del_iter),
                              method='bounded')
        eps_iter = res.x
        budget_SC = [eps_iter, del_iter, total_eps_MA]
        print budget_SC

        """save results"""
        method = 'privacy_budget_SC_S=%s_K=%s_sigma=%s' % (S,K,int(sigma))
        # np.save(method + '.npy', budget_SC)
        scipy.io.savemat(method, dict(budget_SC=budget_SC))



