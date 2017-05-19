import cPickle, sys
import numpy as n
import wikirandom

""" Downloads a bunch of random Wikipedia articles """

# user_args = sys.argv[1:]
# D, seednum = user_args # len(user_args) had better be 2
# numpy.random.seed(int(seednum))

# D = 200000
# N = 1000
D = 400000
# D = 100
howmanysnum = 0
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

    #
    # """ Save the file """
    the_filename = 'wiki_docs_D=%s' %(D)
    with open(the_filename, 'wb') as f:
        cPickle.dump(docset, f)

    the_filename = the_filename+'_title'
    with open(the_filename, 'wb') as f_title:
        cPickle.dump(articlenames, f_title)


