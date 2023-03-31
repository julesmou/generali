import matplotlib.pyplot as plt
from cluster import *
import numpy as np
from scipy.optimize import minimize
from random import *
import matplotlib.pyplot as plt
from slsqp import *
risques = []
revalorisation = []
gains=[]
for i in range (10):
    x0_perturbed = x0 + np.random.uniform(-0.01, 0.01, size=len(x0))
    result = minimize(fun, x0_perturbed, args=(pcc, prime_profit, a, b), method='SLSQP', bounds=bounds, constraints=constraints)
    revalorisation.append(np.mean(result.x))
    risques.append(retention((result.x))[1])
    gains.append(retention(-(result.fun)))

plt.plot(risques, revalorisation)
plt.xlabel("Probabilité de rétention moyenne")
plt.ylabel("Revalorisation")
plt.title("Revalorisation moyenne en fonction de la probabilité de rétention moyenne")
plt.show()


plt.plot(risques, gains)
plt.xlabel("Probabilité de rétention moyenne")
plt.ylabel("Gain")
plt.title("Gain moyenne en fonction de la probabilité de rétention moyenne")
plt.show()