# __author__ = 'mijung'
import cPickle
import os
Data_PATH = "/".join([os.getenv("HOME"), "LDA_data/"])

docset = []
for whichdoc in range(1, 21):
    the_filename = Data_PATH+'wikidata_seednum=_%s' %(whichdoc)
    with open(the_filename, 'rb') as f:
        docset1 = cPickle.load(f)
        docset = docset + docset1
        print "docset %s is loaded" %(whichdoc)

print "docset all loaded"

the_filename = Data_PATH+'wiki_data'
with open(the_filename, 'wb') as f:
    cPickle.dump(docset, f)


