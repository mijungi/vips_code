# wikirandom.py: Functions for downloading random articles from Wikipedia
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


""" mijung's comments """
# (1) I added a few lines starting from line 83: """ Need to do some pre-processing such that each document has less
# than maximum length N """
# (2) You might need to modify get_random_wikipedia_article, depending on how wikipedia forms their articles!
#     It seems like the article formulations change over time. (I also had to change it when I download wiki pages in
#     2016)

# import numpy as n
import random
import re
import string
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


def get_random_wikipedia_article():
	"""
	Downloads a randomly selected Wikipedia article (via
	http://en.wikipedia.org/wiki/Special:Random) and strips out (most
	of) the formatting, links, etc.

	This function is a bit simpler and less robust than the code that
	was used for the experiments in "Online VB for LDA."
	"""
	failed = True
	while failed:
		articletitle = None
		failed = False
		try:
			req = urllib.request.Request('http://en.wikipedia.org/wiki/Special:Random', None, {'User-Agent': 'x'})
			f = urllib.request.urlopen(req)
			line = f.read()

			# result = re.search(r'<title>(.*) - Wikipedia, the free encyclopedia</title>\n', line)
			result = re.search(r'<title>(.*) - Wikipedia</title>\n', line)  # JF: Changed for 2018 Wikipedia
			# formatting.
			articletitle = result.group(1)
			articletitle = articletitle.replace(' ', '_')  # JF: for some reason it doesn't work unless I do this
			print(articletitle)  # JF: added this for debug, but it's actually pretty interesting to see

			req = urllib.request.Request('http://en.wikipedia.org/w/index.php?title=Special:Export/%s&action=submit' \
			                             % (articletitle),
			                             None, {'User-Agent': 'x'})
			f = urllib.request.urlopen(req)
			all = f.read()
		except (urllib.error.HTTPError, urllib.error.URLError):
			print('oops. there was a failure downloading %s. retrying...' \
			      % articletitle)
			failed = True
			continue
		# print 'downloaded %s. parsing...' % articletitle

		try:
			all = re.search(r'<text.*?>(.*)</text', all, flags=re.DOTALL).group(1)
			all = re.sub(r'\n', ' ', all)
			all = re.sub(r'\{\{.*?\}\}', r'', all)
			all = re.sub(r'\[\[Category:.*', '', all)
			all = re.sub(r'==\s*[Ss]ource\s*==.*', '', all)
			all = re.sub(r'==\s*[Rr]eferences\s*==.*', '', all)
			all = re.sub(r'==\s*[Ee]xternal [Ll]inks\s*==.*', '', all)
			all = re.sub(r'==\s*[Ee]xternal [Ll]inks and [Rr]eferences==\s*', '', all)
			all = re.sub(r'==\s*[Ss]ee [Aa]lso\s*==.*', '', all)
			all = re.sub(r'http://[^\s]*', '', all)
			all = re.sub(r'\[\[Image:.*?\]\]', '', all)
			all = re.sub(r'Image:.*?\|', '', all)
			all = re.sub(r'\[\[.*?\|*([^\|]*?)\]\]', r'\1', all)
			all = re.sub(r'\&lt;.*?&gt;', '', all)
		except:
			# Something went wrong, try again. (This is bad coding practice.)
			print('oops. there was a failure parsing %s. retrying...' \
			      % articletitle)
			failed = True
			continue

		""" Need to do some pre-processing such that each document has less than maximum length N """
		# JF: We will do this pre-processing in the download_wikipage.py script instead, since we will need the vocab
		# in the new version, and I don't want to have to reload it, change the function's interface, or pass it
	# around.
		# maxLen = 10000
		# if len(all)>maxLen:
		# print ('word count is above %s' % maxLen)
		#    l = list(all.split()) # converting string to list
		#    all = ' '.join([str(w) for w in random.sample(l, len(l))]) # randomly sample words without replacement
		#    all = all[0:maxLen] # truncate

	return (all, articletitle)


class WikiThread(threading.Thread):
	articles = list()
	articlenames = list()
	lock = threading.Lock()

	def run(self):
		(article, articlename) = get_random_wikipedia_article()
		WikiThread.lock.acquire()
		WikiThread.articles.append(article)
		WikiThread.articlenames.append(articlename)
		WikiThread.lock.release()


def get_random_wikipedia_articles(n):
	"""
	Downloads n articles in parallel from Wikipedia and returns lists
	of their names and contents. Much faster than calling
	get_random_wikipedia_article() serially.
	"""
	maxthreads = 100
	WikiThread.articles = list()
	WikiThread.articlenames = list()
	wtlist = list()
	for i in range(0, n, maxthreads):
		print('downloaded %d/%d articles...' % (i, n))
		for j in range(i, min(i + maxthreads, n)):
			wtlist.append(WikiThread())
			wtlist[len(wtlist) - 1].start()
		for j in range(i, min(i + maxthreads, n)):
			wtlist[j].join()
	return (WikiThread.articles, WikiThread.articlenames)


if __name__ == '__main__':
	t0 = time.time()

	(articles, articlenames) = get_random_wikipedia_articles(10)
	for i in range(0, len(articles)):
		print(articlenames[i])

	t1 = time.time()
	print('took %f' % (t1 - t0))
