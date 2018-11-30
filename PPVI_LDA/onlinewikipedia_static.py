# onlinewikipedia.py: Demonstrates the use of online VB for LDA to
# analyze a bunch of random Wikipedia articles.
#
# Copyright (C) 2010  Matthew D. Hoffman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Mijung's comment on modification:
"""
""" Mijung edited Matt's online LDA code for private Batch variational inference for LDA, Aug 11, 2016 """

import cPickle, string, numpy, getopt, sys, random, time, re, pprint
import numpy as np
import onlineldavb
# import cal_amp_eps
from scipy.optimize import minimize_scalar
import wikirandom
import os
import calculate_privacy_burned as cal_pri

#Data_PATH = "/".join([os.getenv("HOME"), "LDA_data/"])
#Results_PATH = "/".join([os.getenv("HOME"), "LDA_results2/"])
from os.path import expanduser
home = expanduser("~") #JF: makes the above lines work on Windows
Data_PATH = os.path.join(home, "LDA_data")
Results_PATH = os.path.join(home, "LDA_result")

resampleShortDocs = True;

# numpy.random.seed(12345)

def main():

    # unpack input arguments
    # seednum = 1
    # documentstoanalyze  = 2000
    # batchsize = 10
    # priv = 1
    # epsilon = 1
    # comp = 2
    # mech = 0

    seednum = int(sys.argv[1])
    documentstoanalyze = int(sys.argv[2])
    batchsize = int(sys.argv[3])
    priv = int(sys.argv[4]) # 1 is private version, 0 is nonprivate version
    # epsilon = float(sys.argv[5]) # total privacy budget
    comp = int(sys.argv[5]) #
    mech = int(sys.argv[6]) # 0 for Gaussian, 1 for Laplace

    # The number of topics
    #K = 100
    K = 50 #JF

    # load data
    # the_filename = Data_PATH+'wiki_docsmallset'
    # with open(the_filename, 'rb') as f:
    #     docset = cPickle.load(f)

    #the_filename = Data_PATH+'wiki_docsmallset_D=%s' %(400000)
    the_filename = os.path.join(Data_PATH, 'wiki_docsmallset_D=%s' %(400000)) #JF: Make this work on Windows
    if resampleShortDocs:
        the_filename= the_filename + '_resample_short_docs'
    with open(the_filename, 'rb') as f:
        docset = cPickle.load(f)

    D = len(docset)
    print 'document length: %s'%(D)

    nu = batchsize/float(D) # sampling rate
    numpy.random.seed(seednum)

    print 'seednum %s mini-batchsize %s and number of iter %s' %(seednum, batchsize, documentstoanalyze)

    # Our vocabulary
    vocab = file('./dictnostops.txt').readlines()
    W = len(vocab)

    """ privacy budget calculation """
    # (1) to set the same level of burned privacy, we first calculate MA composition
    #sigma = 1.00000000000000000000000000000000000001 #a small value to minimize the noise
    #sigma = 1.1  #an intermediate value
    sigma = 1.24 #an intermediate value
    #sigma = 1.5  #an intermediate value
    #sigma = 2 #a larger value, expected to substantially reduce privacy and performance.


    total_del = 1e-4
    J = documentstoanalyze
    total_eps_MA = cal_pri.moments_accountant(sigma, total_del, nu, J)
    print 'total privacy loss is %f' %(total_eps_MA)

    #(2) strong composition
    del_iter = 1e-6
    res = minimize_scalar(cal_pri.strong_composition, bounds=(0, 50), args=(total_eps_MA, total_del, J, nu, del_iter), method='bounded')
    eps_iter = res.x

    gamma_noise = 0 # we don't use this at all. 

    if comp==0: #MA
        c2 = 2 * np.log(1.25 / del_iter)
        eps_iter = np.sqrt(c2)/sigma
        budget = [eps_iter, del_iter]

    elif comp==1: #strong composition
        budget = [eps_iter, del_iter]
    else:
        print "we don't support this composition"

    if priv:
        print 'private version'

    olda = onlineldavb.OnlineLDA(vocab, K, D, 1./K, 1./K, 1024., 0.7, priv, budget, gamma_noise, mech)


    perplexity = numpy.zeros(documentstoanalyze)

    # for iteration in range(0, maxIter):
    for iteration in range(0, documentstoanalyze):
        # subset of data
        rand_perm_nums =  numpy.random.permutation(len(docset))
        idx_minibatch = rand_perm_nums[0:batchsize]
        docsubset = list(docset[i] for i in idx_minibatch)

        # Give them to online LDA
        (gamma, bound) = olda.update_lambda_docs(docsubset)
        # Compute an estimate of held-out perplexity
        (wordids, wordcts) = onlineldavb.parse_doc_list(docsubset, olda._vocab)
        perwordbound = bound * len(docsubset) / (D * sum(map(sum, wordcts)))
        print '%d:  rho_t = %f,  held-out perplexity estimate = %f' % \
            (iteration, olda._rhot, numpy.exp(-perwordbound))

        perplexity[iteration] = numpy.exp(-perwordbound)


    # save perplexity
    if priv:
        # if gamma_noise:
        #     method = 'private_epsilon_%s_cdp_%s_gamma_noise_%s' % (epsilon, cdp, gamma_noise)
        # else:
        #     method = 'private_epsilon_%s_cdp_%s' %(epsilon, cdp)
        # method = 'static_private_seed=%s_J=%s_S=%s_priv=%s_epsilon=%s_compo=%s_D=%s' %(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], D)

        #method = Results_PATH+'static_private_seed=%s_J=%s_S=%s_priv=%s_epsilon=%s_compo=%s_Lap=%s_D=%s' %(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], total_eps_MA, sys.argv[5], sys.argv[6], D)
        method = os.path.join(Results_PATH,'static_private_seed=%s_J=%s_S=%s_priv=%s_epsilon=%s_compo=%s_Lap=%s_D=%s' %(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], total_eps_MA, sys.argv[5], sys.argv[6], D))
        
    else:
        #method = Results_PATH+'static_nonprivate_seed=%s_J=%s_S=%s_priv=%s_D=%s' %(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], D)
        method = os.path.join(Results_PATH,'static_nonprivate_seed=%s_J=%s_S=%s_priv=%s_D=%s' %(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], D))
    if resampleShortDocs:
        method = method + '_resample_short_docs'
    numpy.save(method+'.npy', perplexity)
    # method = 'private_epsilon_1'
    # filename = method+'_D=_%s_S=_%s' %(D, batchsize)
    # numpy.save(filename+'.npy', test_log_likelihood)

    # save lambda and gamma
    numpy.savetxt(method+'_lambda.dat', olda._lambda)
    numpy.savetxt(method+'_gamma.dat', gamma)

if __name__ == '__main__':
    main()
