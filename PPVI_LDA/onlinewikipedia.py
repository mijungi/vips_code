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

import getopt
import numpy
import os
import pickle
import pprint
import random
import re
import string
import sys
import time

import onlineldavb
import wikirandom

Data_PATH = "/".join([os.getenv("HOME"), "LDA_data/"])


# numpy.random.seed(12345)

def main():
	# unpack input arguments
	# seednum = 1
	# documentstoanalyze  = 2000
	# batchsize = 1000
	# priv = 0
	# epsilon = 1
	# comp = 2

	seednum = int(sys.argv[1])
	documentstoanalyze = int(sys.argv[2])
	batchsize = int(sys.argv[3])
	priv = int(sys.argv[4])  # 1 is private version, 0 is nonprivate version
	epsilon = float(sys.argv[5])  # total privacy budget
	comp = int(sys.argv[6])  # 0 conventional, 1 advanced, 2 CDP

	# The number of topics
	K = 100
	# D = 1000000
	D = 5000000

	nu = batchsize / float(D)  # sampling rate
	numpy.random.seed(seednum)

	print('seednum %s mini-batchsize %s and number of iter %s' % (seednum, batchsize, documentstoanalyze))

	# Our vocabulary
	vocab = file('./dictnostops.txt').readlines()
	W = len(vocab)

	gamma_noise = 0  # will use Laplace noise all the time

	if comp == 2:
		# budget = numpy.sqrt(epsilon/float(documentstoanalyze))
		# budget = numpy.sqrt(epsilon*D/float(2*batchsize))
		budget = numpy.sqrt(2 * epsilon) / float(2 * nu * numpy.sqrt(documentstoanalyze))
	elif comp == 1:
		delta = 0.000001
		budget = epsilon / float(4 * nu * numpy.sqrt(2 * documentstoanalyze * numpy.log(1 / delta)))
	else:
		# budget = epsilon/float(documentstoanalyze)
		budget = epsilon / float(2 * documentstoanalyze * nu)

	if priv:
		print('private version')

	olda = onlineldavb.OnlineLDA(vocab, K, D, 1. / K, 1. / K, 1024., 0.7, priv, budget, gamma_noise)

	# the_filename = Data_PATH+'wiki_data'
	# with open(the_filename, 'rb') as f:
	#     docset = cPickle.load(f)

	# load all the documents
	# docset = []
	# for whichdoc in range(1, 21):
	#     the_filename = Data_PATH+'wikidata_seednum=_%s' %(whichdoc)
	#     with open(the_filename, 'rb') as f:
	#         docset1 = cPickle.load(f)
	#         docset = docset + docset1
	#         print "docset %s is loaded" %(whichdoc)
	#
	# print "docset all loaded"

	perplexity = numpy.zeros(documentstoanalyze)
	# D_test = 10000

	# for iteration in range(0, maxIter):
	for iteration in range(0, documentstoanalyze):
		# subset of data
		# rand_perm_nums =  numpy.random.permutation(len(docset))
		# idx_minibatch = rand_perm_nums[0:batchsize]
		# docsubset = list(docset[i] for i in idx_minibatch)

		# Download some articles
		(docset, articlenames) = \
			wikirandom.get_random_wikipedia_articles(batchsize)
		# Give them to online LDA
		(gamma, bound) = olda.update_lambda_docs(docset)
		# Compute an estimate of held-out perplexity
		(wordids, wordcts) = onlineldavb.parse_doc_list(docset, olda._vocab)
		perwordbound = bound * len(docset) / (D * sum(map(sum, wordcts)))
		print('%d:  rho_t = %f,  held-out perplexity estimate = %f' % \
		      (iteration, olda._rhot, numpy.exp(-perwordbound)))

		# # Give them to online LDA
		# (gamma, bound) = olda.update_lambda_docs(docsubset)
		# # Compute an estimate of held-out perplexity
		# (wordids, wordcts) = onlineldavb.parse_doc_list(docsubset, olda._vocab)
		# perwordbound = bound * len(docsubset) / (D * sum(map(sum, wordcts)))
		# print '%d:  rho_t = %f,  training perplexity estimate = %f' % \
		#     (iteration, olda._rhot, numpy.exp(-perwordbound))

		# compute test perplexity
		# idx_test = rand_perm_nums[batchsize+1:batchsize+1+D_test]
		# doctest = list(docset[i] for i in idx_test)
		#
		# (gamma_test, ss) = olda.do_e_step_docs(doctest)
		# # Estimate held-out likelihood for current values of lambda.
		# bound_test = olda.approx_bound_docs(doctest, gamma_test)
		# (wordids, wordcts_test) = onlineldavb.parse_doc_list(doctest, olda._vocab)
		#
		# # perwordbound_test = bound_test*D_test / float(D*sum(map(sum, wordcts_test)))
		# perword_test_log_likelihood = bound_test / float(sum(map(sum, wordcts_test)))
		# print '%d:  rho_t = %f,  test perplexity estimate = %f' % \
		#     (iteration, olda._rhot, perword_test_log_likelihood)

		perplexity[iteration] = numpy.exp(-perwordbound)

	# save perplexity
	if priv:
		# if gamma_noise:
		#     method = 'private_epsilon_%s_cdp_%s_gamma_noise_%s' % (epsilon, cdp, gamma_noise)
		# else:
		#     method = 'private_epsilon_%s_cdp_%s' %(epsilon, cdp)
		method = 'private_seed=%s_J=%s_S=%s_priv=%s_epsilon=%s_compo=%s' % (
		sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
	else:
		method = 'Nonprivate_seed=%s_J=%s_S=%s_priv=%s_epsilon=%s_compo=%s' % (
		sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])

	numpy.save(method + '.npy', perplexity)
	# method = 'private_epsilon_1'
	# filename = method+'_D=_%s_S=_%s' %(D, batchsize)
	# numpy.save(filename+'.npy', test_log_likelihood)

	# save lambda and gamma
	numpy.savetxt(method + '_lambda.dat', olda._lambda)
	numpy.savetxt(method + '_gamma.dat', gamma)


if __name__ == '__main__':
	main()
