import matplotlib.pyplot as plt
import numpy as np
import random
import time
import pandas as pd
import math


# Charger les données à partir du fichier Excel
data = pd.read_csv(
    "/Users/antbe/Downloads/ST7_projet/generali/filtre_0.01_avec_augmentation_50.csv")


X = data[['prime_profit', 'pcc', 'coeff_non_prix',
          'coeff_prix']]


# on récupère la longueur de la liste
taille_individu = len(X[['coeff_non_prix']])

# on ajoute un bruit blanc d'écart-type = 10% de la moyenne au coeff non prix
#mean_a = X[['coeff_non_prix']].sum()/taille_individu
#Gaussien_a = np.random.normal(0, 0.1*np.abs(mean_a), taille_individu)
#X['coeff_non_prix'] = X['coeff_non_prix'] + Gaussien_a

# on ajoute un bruit blanc d'écart-type = 10% de la moyenne au coeff prix
#mean_b = X[['coeff_prix']].sum()/taille_individu
#Gaussien_b = np.random.normal(0, 0.1*np.abs(mean_b), taille_individu)
#X['coeff_prix'] = X['coeff_prix'] + Gaussien_b


def retention(x):
    ret = 1/(1+np.exp(X["coeff_non_prix"] + X["coeff_prix"]*x))
    somme_ret = ret.sum()
    mean_rate = somme_ret/taille_individu
    return ret, mean_rate


def fun(x):
    ret = retention(x)[0]
    marge = (X["prime_profit"] * (1+x)-X['pcc'])*ret
    res = marge.sum()
    return res


# Problem settings
D = taille_individu  # Dimension of the search space
m = -0.05 * np.ones(D)  # Lower bound
M = 0.1 * np.ones(D)  # Upper bound


prob = fun

# Parameters for DE
N = 50  # Population Size
G = 50  # No. of iterations
F = 0.8  # Scaling factor
C = 0.7  # Crossover probability


# Starting of DE
f = np.empty((N))  # Collection of vectors to store f(population_g)
P = np.empty((N, D))  # Collection of matrices to store P(population_g)
# Collection of matrices to store the mutant vector (V(g+1))
V = np.empty((N, D))
# Collection of matrices to store the trial solutions (U(g+1))
U = np.empty((N, D))

Pinit = np.random.rand(N, D)*(M-m) + m  # Initial P

Fplot = []
Ret = []

for n in range(N):
    f[n] = prob(Pinit[n])  # Evaluating f(population_1)

P = Pinit


# Boucle principale
tic = time.time()
for g in range(G-1):
    for n in range(N):
        # Sélection de r1, r2, r3 distincts de la population
        while True:
            r1 = np.random.randint(N)
            if r1 != n:
                break
        while True:
            r2 = np.random.randint(N)
            if r2 != n and r2 != r1:
                break
        while True:
            r3 = np.random.randint(N)
            if r3 != n and r3 != r2 and r3 != r1:
                break

        # Mutation
        V[n] = P[r1] + F*(P[r2]-P[r3])

        # Croisement
        if np.random.rand() <= C:
            U[n] = V[n]
        else:
            U[n] = P[n]

        # Sélection
        f_U = prob(U[n])
        if f_U >= f[n]:
            P[n] = U[n]
            f[n] = f_U

        # préparation du graphique
    Fplot.append([np.min(f, axis=0), np.max(f, axis=0),
                  np.median(f, axis=0), np.mean(f, axis=0)])
    opt_idx = np.argmin(f[-1])
    opt_sol = P[opt_idx]
    Ret.append(retention(opt_sol)[1])


# Temps de calcul
calculation_time = time.time() - tic

# Résultats clés
print('=========== OPTIMAL SOLUTION')
print(f'calculation time = {calculation_time}s')
opt_val, opt_idx = np.min(f[-1]), np.argmin(f[-1])
opt_sol = P[opt_idx]
print('=========== OPTIMAL SOLUTION ===========')
print('The optimal solution x* is:')
print(opt_sol)
print(f'The corresponding objective value is: {opt_val}')
opt_ret = retention(opt_sol)[1]
print(f'le taux de rétention moyen de la valeur optimale est :{opt_ret}%')

# Analyse comportementale

# Critère

# on crée fmin,fmax,fmedian et fmean
fmin = []
for i in range(len(Fplot)):
    fmin.append(Fplot[i][0])

fmax = []
for i in range(len(Fplot)):
    fmax.append(Fplot[i][1])

fmedian = []
for i in range(len(Fplot)):
    fmedian.append(Fplot[i][2])

fmean = []
for i in range(len(Fplot)):
    fmean.append(Fplot[i][3])

# on les traces
index = list(range(1, len(Fplot)+1))

plt.plot(index, fmin, 'r.', index, fmax, 'r.',
         index, fmedian, 'm', index, fmean, 'b')
plt.legend(['min', 'max', 'median', 'mean'])
plt.title('CRITERION')
plt.grid(True)
plt.show()

plt.plot(index, Ret)
plt.legend(['taux de rétention'])
plt.title('CRITERION')
plt.grid(True)
plt.show()
