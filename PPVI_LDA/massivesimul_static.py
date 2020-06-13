# __author__ = 'mijung'

import subprocess
import sys

D = 400000
# Smat = [10, 20, 50, 100, 200, 400]
# Smat = [10, 100, 200]
Smat = [20000]  # JF
# Jmat = [0.05, 0.1, 0.2, 0.4]
# Jmat = [0.4]
Jmat = [1]
seednummat = [0]
# seednummat = [4, 5, 6, 7, 8, 9, 10]
# unpack inputs
priv = sys.argv[1]  # 1 for private, 0 for non-private
comp = sys.argv[2]  # 0 for MA, 1 for SC
mech = sys.argv[3]  # 0 for Gaussian, 1 for Laplace
# priv = 1
# comp = 0 # 0 for MA composition, 1 for strong composition
# mech = 0 # 0 for Gaussian, 1 for Laplace


# run this code for two different scenarios
# (1) MA composition
# (2) strong composition


for seednum in seednummat:
	for S in Smat:
		for Jlist in Jmat:
			J = int(Jlist * D / float(S))
			print([seednum, J, S, priv, comp, mech])
			sys.argv = ['onlinewikipedia_static.py', str(seednum), str(J), str(S), str(priv), str(comp), str(mech)]
			exec(compile(open('onlinewikipedia_static.py', "rb").read(), 'onlinewikipedia_static.py', 'exec'))
