from cluster import *
import numpy as np
from scipy.optimize import minimize
from random import *
n = 100
# Définir la plage de valeurs pour chaque variable
#r = 1/(1+np.exp(args[2][i]+args[3][i]*x1>><<<<<<<<<<>[i]))
#plt.xlabel('Marge esperée')
#plt.ylabel('Probabilité de rétention')
#plt.title('Frontière efficiente')
#plt.show()
# a et b sont les paramètres non prix et prix pour chaque persone, pcc et prime profit les primes payées actuellement
chemin_fichier = "/Users/julesmouradian/Desktop/Centrale/donnees.generali.xltx"
# Lecture du fichier Excel contenant les données
df = pd.read_excel(chemin_fichier)

# Sélection des colonnes à utiliser pour le clustering
Y= df[['prime_profit','pcc','coeff_non_prix', 'coeff_prix']]
# Standardiser les données pour avoir une moyenne nulle et une variance unitaire
newX_scaled = pd.DataFrame(X)
newY= pd.DataFrame(Y)
# Supprimer les lignes contenant des valeurs manquantes
newX_scaled = newX_scaled.dropna()
newY=newY.dropna()
b=[]
a=[]
pcc=[]
prime_profit=[]  

for i in range(100):
    a.append(Y['coeff_non_prix'][i])
    b.append(Y['coeff_prix'][i])
    pcc.append(Y['pcc']['i'])
    prime_profit.append(Y['prime_profit'][i])
args = [pcc,prime_profit, a, b]


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
        res = res+(prime_profit[i]*(1+x[i])-pcc[i])*ret[i]
    return res
# Définir les contraintes (sous forme d'un dictionnaire)'''
constraints = []
for i in range(n):
    constraints.append({'type': 'ineq', 'fun': lambda x, i=i:  x[i] + 0.1})
    constraints.append({'type': 'ineq', 'fun': lambda x, i=i:   -x[i] + 0.1})
constraints.append({'type': 'ineq', 'fun': lambda x:  retention(x)[1] - 0.95})

# Définir la borne inférieure et supérieure de chaque variable
bounds = tuple((None, None) for _ in range(n))

# Définir la valeur initiale
l=9
x0 = np.array(l*[random()/10-0.1])
x1_vals = np.linspace(-1, 1, 100)
x2_vals = np.linspace(-1, 1, 100)

result = minimize(fun, x1_vals, args=(pcc, prime_profit, a,b), method='SLSQP', bounds=bounds, constraints=constraints)
for i in range (100):
    result['x'][i]
    