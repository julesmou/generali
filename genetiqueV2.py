import matplotlib.pyplot as plt
import numpy as np
import random
import time
import pandas as pd
import math


chemin_fichier = "/Users/julesmouradian/Desktop/Centrale/donnees.generali.xltx"

# Charger les données à partir du fichier Excel
data = pd.read_excel(chemin_fichier)
data['coeff_prix'] = data['coeff_prix'].multiply(5)

X = data[['prime_profit', 'pcc', 'coeff_non_prix',
          'coeff_prix']]

# Supprimer les lignes contenant des valeurs manquantes
X_scaled = X.dropna()

# on récupère la longueur de la liste
taille_individu = len(X_scaled[['coeff_non_prix']])


def retention(x):
    ret = []
    m = 0
    for i in range(taille_individu):
        r = 1/(1+np.exp(X_scaled["coeff_non_prix"].iloc[i] +
               X_scaled["coeff_prix"].iloc[i]*x[i]))
        ret.append(r)  # on définit la proba de rétention pour chaque client
        m = ret[i]+m
    return ret, m/taille_individu


def fun(x):
    ret = retention(x)[0]
    res = 0
    for i in range(taille_individu):
        res = res+(X_scaled["prime_profit"].iloc[i] *
                   (1+x[i])-X_scaled['pcc'].iloc[i])*ret[i]
    return res


# Problem settings
D = taille_individu  # Dimension of the search space
size_of_the_box = 0.1
m = -size_of_the_box * np.ones(D)  # Lower bound
M = size_of_the_box * np.ones(D)  # Upper bound


prob = fun

# Parameters for DE
N = 100  # Population Size
G = 10  # No. of iterations
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

for n in range(N):
    f[n] = prob(Pinit[n])  # Evaluating f(population_1)

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
        j_rand = np.random.randint(D)
        for j in range(D):
            if np.random.rand() <= C or j == j_rand:
                U[n] = V[n]
            else:
                U[n] = P[n]

        # Sélection
        f_U = prob(U[n])
        if f_U >= f[n]:
            P[n] = U[n]
            f[n] = f_U

# Temps de calcul
calculation_time = time.time() - tic

# Résultats clés
print('=========== OPTIMAL SOLUTION')
print(f'calculation time = {calculation_time}s')
opt_val, opt_idx = np.min(f[:, -1]), np.argmin(f[:, -1])
opt_sol = P[opt_idx, :, G-1]
print('=========== OPTIMAL SOLUTION ===========')
print('The optimal solution x* is:')
print(opt_sol)
print(f'The corresponding objective value is: {opt_val}')

# Analyse comportementale

# Critère
'''plt.plot(range(G), np.min(f, axis=0), 'r.', range(G), np.max(f, axis=0), 'r.',
         range(G), np.median(f, axis=0), 'm', range(G), np.mean(f, axis=0), 'b')
plt.legend(['min', 'max', 'median', 'mean'])
plt.title('CRITERION')
plt.grid(True)

# distance

d2mean = np.empty((N, G))
pop_stats = np.empty((6, G))
for g in range(G):
    # Calculate the average position for the current generation
    avg_pos = np.mean(P[:, :, g], axis=1)
    # Compute the distances between each individual and the average position
    dist_to_avg = np.sqrt(
        np.sum((P[:, :, g] - avg_pos[:, np.newaxis])**2, axis=0))
    # Store the distances in the d2mean table
    d2mean[:, g] = dist_to_avg
    # Calculate population statistics
    pop_stats[:, g] = np.array([np.min(dist_to_avg),
                                np.max(dist_to_avg),
                                np.median(dist_to_avg),
                                np.mean(dist_to_avg),
                                np.median(dist_to_avg) + np.std(dist_to_avg),
                                max(np.median(dist_to_avg) - np.std(dist_to_avg), np.median(dist_to_avg)/100)])

# Plot population statistics
fig, ax = plt.subplots()
ax.plot(np.arange(1, G+1), pop_stats[0], 'r.', label='min')
ax.plot(np.arange(1, G+1), pop_stats[1], 'r.', label='max')
ax.plot(np.arange(1, G+1), pop_stats[2], 'm', label='median')
ax.plot(np.arange(1, G+1), pop_stats[3], 'b', label='mean')
ax.plot(np.arange(1, G+1), pop_stats[4], 'g', label='median+std')
ax.plot(np.arange(1, G+1), pop_stats[5], 'g', label='median-std')
ax.legend()
ax.set_title('POPULATION /mean')
ax.set_xlabel('Generation')
ax.set_ylabel('Distance to average position')
ax.grid(True)
ax.autoscale(True)
plt.show()'''
