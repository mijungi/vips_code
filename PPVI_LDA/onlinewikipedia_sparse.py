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
""" Mijung edited Matt's online LDA code for private Batch variational inference for LDA, Sep 02, 2016 """
""" Using Sparse : adding noise only to those element that's above a random threshold """

import cPickle, string, numpy, getopt, sys, random, time, re, pprint

import onlineldavb_sparse
import cal_amp_eps
from scipy.optimize import minimize_scalar
import wikirandom
import os

Data_PATH = "/".join([os.getenv("HOME"), "LDA_data/"])


def main():

    # unpack input arguments
    seednum = 1
    documentstoanalyze  = 2000
    batchsize = 10
    priv = 1
    epsilon = 1
    comp = 2

    # seednum = int(sys.argv[1])
    # documentstoanalyze = int(sys.argv[2])
    # batchsize = int(sys.argv[3])
    # priv = int(sys.argv[4]) # 1 is private version, 0 is nonprivate version
    # epsilon = float(sys.argv[5]) # total privacy budget
    # comp = int(sys.argv[6]) # 0 conventional, 1 advanced, 2 CDP

    # The number of topics
    K = 100

    # load data
    the_filename = Data_PATH+'wiki_docsmallset'
    with open(the_filename, 'rb') as f:
        docset = cPickle.load(f) # this should be D=200,000

    D = len(docset)
    print 'document length: %s'%(D)

    nu = batchsize/float(D) # sampling rate
    numpy.random.seed(seednum)

    print 'seednum %s mini-batchsize %s and number of iter %s' %(seednum, batchsize, documentstoanalyze)

    # Our vocabulary
    vocab = file('./dictnostops.txt').readlines()
    W = len(vocab)

    gamma_noise = 0 # will use Laplace noise all the time
    res = minimize_scalar(cal_amp_eps.f, bounds=(0, 400), args=(epsilon, comp, documentstoanalyze, nu), method='bounded')
    budget = res.x
    print res.x

    if priv:
        print 'private version'

    olda = onlineldavb_sparse.OnlineLDA(vocab, K, D, 1./K, 1./K, 1024., 0.7, priv, budget, gamma_noise)

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
        (wordids, wordcts) = onlineldavb_sparse.parse_doc_list(docsubset, olda._vocab)
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
        method = 'sparse_private_seed=%s_J=%s_S=%s_priv=%s_epsilon=%s_compo=%s_D=%s' %(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], D)
    else:
        method = 'sparse_nonprivate_seed=%s_J=%s_S=%s_priv=%s_epsilon=%s_compo=%s_D=%s' %(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], D)

    numpy.save(method+'.npy', perplexity)
    # method = 'private_epsilon_1'
    # filename = method+'_D=_%s_S=_%s' %(D, batchsize)
    # numpy.save(filename+'.npy', test_log_likelihood)

    # save lambda and gamma
    numpy.savetxt(method+'_lambda.dat', olda._lambda)
    numpy.savetxt(method+'_gamma.dat', gamma)

if __name__ == '__main__':
    main()
