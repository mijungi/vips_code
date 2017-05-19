# __author__ = 'mijung'

import numpy as np
import matplotlib.pyplot as plt

epsilon_array = [0.1]
k_array = np.linspace(0, 100, 200)

delta = 0.000001

for eps in epsilon_array:
    for k in k_array:
        #(1) traditional composition
        eps_tot = k*eps
        plt.plot(k, eps_tot, '.-', color = (0.0, 0.0, 0.0))

        #(2) advanced composition
        eps_tot = np.sqrt(2*k*np.log(1/delta))*eps + k*eps*(np.exp(eps) - 1)
        plt.plot(k, eps_tot, '.-', color = (0.0, 0.0, 1))

        #(3) CDP composition
        eps_tot = k*(eps**2)/(2.)
        plt.plot(k, eps_tot, '.-', color = (1, 0.0, 0.0))

# plt.xscale('log')
plt.xlabel('# iterations')
plt.ylabel('total privacy budget')
plt.title('Composition comparison (per-iteration budget=0.1)')
plt.show()


