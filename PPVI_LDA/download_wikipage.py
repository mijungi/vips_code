import cPickle, sys
import numpy as n
import wikirandom
import os
import random, re, string

""" Downloads a bunch of random Wikipedia articles """

#Data_PATH = "/".join([os.getenv("HOME"), "LDA_data/"])
from os.path import expanduser
home = expanduser("~") #JF: makes the above line work on Windows
Data_PATH = os.path.join(home, "LDA_data")

""" I will save every 1000 docs (with a different seed number),  
then combine these in save_docset to make a single document set with D=400,000 """

D = 1000
#howmanysnum = 100 # 400,000/1000 = 100
howmanysnum = 400 # 400,000/1000 = 400 #Changed by JF
seednummat = n.arange(0,howmanysnum)
maxLen = 500 #JF: maximum number of words in a document. Need to set this the same in onlineldavb.py, currently line 205.
vocabFilename = './dictnostops.txt' #JF: need the vocab for enforcing the maximum length of a document
resampleShortDocuments = False #JF: whether to resample documents that are shorter than maxLen up to the maximum length.  This could be helpful if short documents are getting swamped by the noise

length_seed = n.shape(seednummat)
print length_seed[0]

#JF: This function is used to enforce that all documents are less than the maximum specified length, taking into account the vocabulary
def enforceDocumentMaxLength(docset, maxLen, vocabFilename, resampleShortDocuments):
    vocab = open(vocabFilename).readlines()
    for i in range(0, len(vocab)):
            vocab[i] = vocab[i].lower()
            vocab[i] = re.sub(r'[^a-z]', '', vocab[i])
    W = len(vocab)
    
    for i in range(0, len(docset)):
        #first do the same preprocessing that onlineldavb.py performs
        docset[i] = docset[i].lower()
        docset[i] = re.sub(r'-', ' ', docset[i])
        docset[i] = re.sub(r'[^a-z ]', '', docset[i])
        docset[i] = re.sub(r' +', ' ', docset[i])
        
        #get only in-vocab words
        words = string.split(docset[i])
        wordsInVocab = []
        for word in words:
            if (word in vocab):
                wordsInVocab.append(word)
        print len(wordsInVocab)
        #check length of document and determine whether to bootstrap resample the words
        if len(wordsInVocab) > maxLen or resampleShortDocuments:
            adjustedWordsInVocab = [];
            print 'resampling to length ' + str(maxLen)
            for j in range(0, maxLen):
                adjustedWordsInVocab.append(random.choice(wordsInVocab)) #random sampling WITH replacement
            wordsInVocab = adjustedWordsInVocab
        docset[i] = ' '.join(wordsInVocab) #create final space-separated pre-processed document
    return docset


for i in range(0, length_seed[0]):
    seednum = seednummat[i]
    print seednum
    n.random.seed(int(seednum))

    # Download some articles
    """ Need to do some pre-processing such that each document has less than maximum length N """
    (docset, articlenames) = wikirandom.get_random_wikipedia_articles(int(D))
    print 'enforcing document length requirement for privacy'
    docset = enforceDocumentMaxLength(docset, maxLen, vocabFilename, resampleShortDocuments) #JF: ensure that all documents are no longer than maxLen

    # """ Save the file """
    #the_filename = Data_PATH+'wiki_docs_seednum=%s' %(seednum)
    the_filename = os.path.join(Data_PATH, 'wiki_docs_seednum=%s' %(seednum))
    with open(the_filename, 'wb') as f:
        cPickle.dump(docset, f)


