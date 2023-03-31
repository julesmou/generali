from cluster import *
import numpy as np
from scipy.optimize import minimize
from random import *
import matplotlib.pyplot as plt
n = 9
#df=df.dropna()
#df['cluster_labels'] = kmeans.labels_
#df.dropna(inplace=True) #supprimer les lignes contenant des valeurs manquantes de df
# a et b sont les paramètres non prix et prix pour chaque persone, pcc et prime profit les primes payées actuellement

b=[]
a=[]
pcc=[]
prime_profit=[]   
for i in range(9):
    a.append(V[i][2][1])
    b.append(V[i][2][2])
    pcc.append(V[i][1])
    prime_profit.append(V[i][0])
args = [pcc,prime_profit, a, b]
#print(args)

def retention(x):
    ret = []
    m=0
    r=0
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
        res = res-(prime_profit[i]*(1+x[i])-pcc[i])*ret[i]
    return res
# Définir les contraintes (sous forme d'un dictionnaire)'''


constraints = []
for i in range(n):
    constraints.append({'type': 'ineq', 'fun': lambda x, i=i:  x[i] + 0.05})
    constraints.append({'type': 'ineq', 'fun': lambda x, i=i:   -x[i] + 0.1})
constraints.append({'type': 'ineq', 'fun': lambda x:  - retention(x)[1] + 0.91})

# Définir la borne inférieure et supérieure de chaque variable
bounds = tuple((-0.1, 0.1) for _ in range(n))
x0 = np.random.uniform(-0.05, 0.1, n)
# Définir la valeur initiale
l=9
#x0 = np.array(l*[2*(random()/10-0.08)])
# Minimiser la fonction objectif en utilisant la méthode SLSQP

D = {}
best_result = None
best_fun_value = float('inf')

result = minimize(fun, x0, args=(pcc, prime_profit, a, b), method='SLSQP', bounds=bounds, constraints=constraints)

revalorisations = []

# Exécuter l'optimisation 100 fois et stocker les valeurs de revalorisation moyenne
for _ in range(100):
    x0 = np.random.uniform(-0.05, 0.1, n)
    result = minimize(fun, x0, args=(pcc, prime_profit, a, b), method='SLSQP', bounds=bounds, constraints=constraints)
    revalorisations.append(np.mean(result.x))

# Afficher la distribution des revalorisations moyennes
plt.hist(revalorisations, bins=20, density=True, alpha=0.6, color='b')
plt.xlabel('Revalorisation moyenne')
plt.ylabel('Probabilité')
plt.title('Distribution des revalorisations moyennes à 0,91')
plt.show()