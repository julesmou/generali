from scipy import optimize
from data import *
import numpy as np
from scipy.optimize import minimize
from random import *
n = 4

# a et b sont les paramètres non prix et prix pour chaque persone, pcc et prime profit les primes payées actuellement
args = (pcc, prime_profit, a, b)


def retention(x):
    ret = []
    m=0
    for i in range(n):
        r = 1/(1+np.exp(args[2][i]+args[3][i]*x[i]))
        ret.append(r)  # on définit la proba de rétention pour chaque client
    for i in range (n):
        m=ret(i)+m
        
    return ret,m/n
        

def fun(x, args):
    ret = retention(x)[0]
    res = 0
    for i in range(n):
        res = res+(args[1][i](1+x[i])-args[0][i])*ret[i]
    return res

# Définir les contraintes (sous forme d'un dictionnaire)
constraints = ([{'type': 'ineq', 'fun': lambda x:  x[i] + 0.1} for i in range(n)],
               {'type': 'ineq', 'fun': lambda x:  retention(x)[1] - 0.95})

# Définir la borne inférieure et supérieure de chaque variable
bounds = ((None, None), (None, None))

# Définir la valeur initiale
l=4
x0 = np.array(l*[random()/10-0,1])
print(x0)
# Minimiser la fonction objectif en utilisant la méthode SLSQP
result = minimize(fun, x0, method='SLSQP', bounds=bounds, constraints=constraints)

# Afficher le résultat
print(result)