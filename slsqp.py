
from data import *
import numpy as np
from scipy.optimize import minimize
from random import *
n = 4

# a et b sont les paramètres non prix et prix pour chaque persone, pcc et prime profit les primes payées actuellement
args = [pcc, prime_profit, a, b]


def retention(x):
    ret = []
    m=0
    for i in range(n):
        r = 1/(1+np.exp(args[2][i]+args[3][i]*x[i]))
        ret.append(r)  # on définit la proba de rétention pour chaque client
    for i in range (n):
        m=ret[i]+m
        
    return ret,m/n
        

def fun(x, *args):
    pcc, prime_profit, a, b = args
    ret = retention(x)[0]
    res = 0
    for i in range(n):
        res = res+(prime_profit[i]*(1+x[i])-pcc[i])*ret[i]
    return res

# Définir les contraintes (sous forme d'un dictionnaire)
constraints = [{'type': 'ineq', 'fun': lambda x, i=i:  x[i] + 0.1} for i in range(n)]
constraints.append({'type': 'ineq', 'fun': lambda x:  retention(x)[1] - 0.95})

# Définir la borne inférieure et supérieure de chaque variable
bounds = tuple((None, None) for _ in range(n))

# Définir la valeur initiale
l=4
x0 = np.array(l*[random()/10-0.1])
# Minimiser la fonction objectif en utilisant la méthode SLSQP
result = minimize(fun, x0, args=(pcc, prime_profit, a, b), method='SLSQP', bounds=bounds, constraints=constraints)

# Afficher le résultat
print(result)