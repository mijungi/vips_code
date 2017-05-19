# __author__ = 'mijung'

import sys
import subprocess
# sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] = seednum, priv, eps, comp

# maxseednum =
# D = 50000
# D = 200000
# D = 200000
# D = 5000000
# Smat = [200, 800, 3200, 12800]
# Jmat = [0.01, 0.05, 0.1]
# Smat = [10]
# D = 200000
D = 400000
# D = 800000
# D = 1600000
Smat = [10, 20, 50, 100, 200, 400]
Jmat = [0.05, 0.1, 0.2, 0.4]
# Jmat = [0.4]
# Jmat = [0.2]
# Jmat = [0.05]
seednummat = [4, 5, 6, 7, 8, 9, 10]
# unpack inputs
# seednum = sys.argv[1]
priv = sys.argv[1]
eps = sys.argv[2]
comp = sys.argv[3]
mech = sys.argv[4] # 0 for Gaussian, 1 for Laplace

# for seednum in range(0, maxseednum):
    # variables = [seednum, J, S, priv, eps, comp]
for seednum in seednummat:
    for S in Smat:
        for Jlist in Jmat:
            J = int(Jlist*D/float(S))
            print [seednum, J, S, priv, eps, comp, mech]
            sys.argv = ['onlinewikipedia_static.py', str(seednum), str(J), str(S),  str(priv), str(eps), str(comp), str(mech)]
            execfile('onlinewikipedia_static.py')
