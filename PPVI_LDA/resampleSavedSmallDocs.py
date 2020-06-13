# __author__ = 'Jimmy'
# This script resamples documents that are less than maxLength so that they are exactly maxLength.
# It expects that longer documents have been resampled to maxLength (though it will still work
# otherwise, but it specifically doesn't resample documents that are exactly the correct length,
# under the understanding that these are longer docs that have already been resampled)
import os
import pickle
import random
import re
import string
from os.path import expanduser

home = expanduser("~")  # JF: makes the above line work on Windows
Data_PATH = os.path.join(home, "LDA_data")

maxLen = 500  # maximum number of words in a document. Need to set this the same in onlineldavb.py, currently line 205.
vocabFilename = './dictnostops.txt'  # need the vocab for enforcing the maximum length of a document
resampleShortDocuments = True  # whether to resample documents that are shorter than maxLen up to the maximum length.


#  This could be helpful if short documents are getting swamped by the noise


# JF: This function is used to enforce that all documents are less than the maximumspecified length, taking into
# account the vocabulary
def enforceDocumentMaxLength(docset, maxLen, vocabFilename, resampleShortDocuments):
	vocab = open(vocabFilename).readlines()
	for i in range(0, len(vocab)):
		vocab[i] = vocab[i].lower()
		vocab[i] = re.sub(r'[^a-z]', '', vocab[i])
	W = len(vocab)

	for i in range(0, len(docset)):
		# first do the same preprocessing that onlineldavb.py performs
		docset[i] = docset[i].lower()
		docset[i] = re.sub(r'-', ' ', docset[i])
		docset[i] = re.sub(r'[^a-z ]', '', docset[i])
		docset[i] = re.sub(r' +', ' ', docset[i])

		# get only in-vocab words
		words = string.split(docset[i])
		wordsInVocab = []
		for word in words:
			if (word in vocab):
				wordsInVocab.append(word)
		print(len(wordsInVocab))
		if len(wordsInVocab) == 0:
			docset[i] = ''
			continue;
		# check length of document and determine whether to bootstrap resample the words
		if len(wordsInVocab) > maxLen or (len(wordsInVocab) < maxLen and resampleShortDocuments):  # modified from my
			# other script - only resample up to full length if needed. Skip if it's already been downsampled to maxLen
			adjustedWordsInVocab = [];
			print('resampling to length ' + str(maxLen))
			for j in range(0, maxLen):
				adjustedWordsInVocab.append(random.choice(wordsInVocab))  # random sampling WITH replacement
			wordsInVocab = adjustedWordsInVocab
		docset[i] = ' '.join(wordsInVocab)  # create final space-separated pre-processed document
	return docset


docset = []
for whichdoc in range(0, 400):
	# the_filename = Data_PATH+'wiki_docs_seednum=%s' %(whichdoc)
	the_filename = os.path.join(Data_PATH, 'wiki_docs_seednum=%s' % (whichdoc))
	with open(the_filename, 'rb') as f:
		docset1 = pickle.load(f)
		docset2 = enforceDocumentMaxLength(docset1, maxLen, vocabFilename, resampleShortDocuments)
		print("docset %s is loaded" % (whichdoc))
		the_filename2 = os.path.join(Data_PATH, 'wiki_docs_seednum=%s_resample_short_docs' % (whichdoc))
		with open(the_filename2, 'wb') as f:
			pickle.dump(docset2, f)
