# __author__ = 'mijung'

import pickle
import os
import numpy

Data_PATH = "/".join([os.getenv("HOME"), "LDA_data/"])

seednum = 20
numpy.random.seed(seednum)

the_filename = Data_PATH+'wiki_data'
with open(the_filename, 'rb') as f:
    docset = pickle.load(f)
print("docset all loaded")
print(len(docset))
# rand_perm_nums =  numpy.random.permutation(len(docset))
# # how_many_doc = 200000
# how_many_doc = 800000
# idx_to_take = rand_perm_nums[0:how_many_doc]
# docsmallset = list(docset[i] for i in idx_to_take)
#
# # the_filename = Data_PATH+'wiki_docsmallset' # this was for 200,000
# the_filename = Data_PATH+'wiki_docsmallset_D=%s' %(how_many_doc)
# with open(the_filename, 'wb') as f:
#     cPickle.dump(docsmallset, f)
# print "smaller doc saved"
