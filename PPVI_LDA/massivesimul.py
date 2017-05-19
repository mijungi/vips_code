# __author__ = 'mijung'

import sys
import subprocess
# sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] = seednum, priv, eps, comp

# maxseednum =
# D = 1000000
D = 5000000
Smat = [800, 3200]
#Jmat = [0.01, 0.05, 0.1]
Jmat = [0.05, 0.1]

# unpack inputs
seednum = sys.argv[1]
priv = sys.argv[2]
eps = sys.argv[3]
comp = sys.argv[4]

# for seednum in range(0, maxseednum):
    # variables = [seednum, J, S, priv, eps, comp]
for S in Smat:
    for Jlist in Jmat:
        J = int(Jlist*D/float(S))
        print [seednum, J, S, priv, eps, comp]
        sys.argv = ['onlinewikipedia.py', str(seednum), str(J), str(S),  str(priv), str(eps), str(comp)]
        execfile('onlinewikipedia.py')
