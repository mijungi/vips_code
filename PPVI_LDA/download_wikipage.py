import cPickle, sys
import numpy as n
import wikirandom
import os

""" Downloads a bunch of random Wikipedia articles """

Data_PATH = "/".join([os.getenv("HOME"), "LDA_data/"])

""" I will save every 1000 docs (with a different seed number),  
then combine these in save_docset to make a single document set with D=400,000 """

D = 1000
howmanysnum = 100 # 400,000/1000 = 100
seednummat = n.arange(0,howmanysnum)

length_seed = n.shape(seednummat)
print length_seed[0]

for i in range(0, length_seed[0]):
    seednum = seednummat[i]
    print seednum
    n.random.seed(int(seednum))

    # Download some articles
    """ Need to do some pre-processing such that each document has less than maximum length N """
    (docset, articlenames) = wikirandom.get_random_wikipedia_articles(int(D))

    # """ Save the file """
    the_filename = Data_PATH+'wiki_docs_seednum=%s' %(seednum)
    with open(the_filename, 'wb') as f:
        cPickle.dump(docset, f)



