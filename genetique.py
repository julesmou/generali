#from test import *
import random
import numpy as np
import random as rd
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

chemin_fichier = "/Users/antbe/Downloads/ptf_mrh_2020_elast_plr.xlsx"

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

# Fonction objectif à optimiser


def objectif(x):
    result = fun(x, X_scaled)
    return result


# Définition des paramètres de l'algorithme génétique
taille_population = 100
nb_generations = 50
probabilite_mutation = 0.1  # paramètre à modifier, voir sur le TD de metaheuristic


# Création de la population initiale
population = []
for i in range(taille_population):
    individu = []
    for j in range(taille_individu):
        # Génération aléatoire d'un gène
        gene = (np.random.rand(1)/5-0.1)
        individu.append(gene)
    population.append(individu)

# Boucle principale de l'algorithme génétique
for generation in range(nb_generations):
    # Évaluation de la fonction objectif pour chaque individu de la population
    evaluations = []
    for individu in population:
        evaluations.append(objectif(individu))

    # Sélection des meilleurs individus pour la reproduction
    meilleurs_individus = []
    for i in range(int(taille_population/2)):
        index1 = evaluations.index(min(evaluations))
        evaluations[index1] = float('inf')
        index2 = evaluations.index(min(evaluations))
        evaluations[index2] = float('inf')
        meilleurs_individus.append(population[index1])
        meilleurs_individus.append(population[index2])

    # Reproduction des meilleurs individus pour créer une nouvelle population
    nouvelle_population = []
    for i in range(int(taille_population/2)):
        parent1 = random.choice(meilleurs_individus)
        parent2 = random.choice(meilleurs_individus)
        enfant1 = parent1[:int(taille_individu/2)] + \
            parent2[int(taille_individu/2):]
        enfant2 = parent2[:int(taille_individu/2)] + \
            parent1[int(taille_individu/2):]
        nouvelle_population.append(enfant1)
        nouvelle_population.append(enfant2)

    # Mutation aléatoire des individus de la nouvelle population
    for individu in nouvelle_population:
        for i in range(taille_individu):
            if random.random() < probabilite_mutation:
                individu[i] = -individu[i]

    # Remplacement de l'ancienne population par la nouvelle
    population = nouvelle_population

# Sélection de la meilleure solution trouvée
meilleur_individu = population[0]
meilleure_evaluation = objectif(meilleur_individu)
for individu in population:
    evaluation = objectif(individu)
    if evaluation < meilleure_evaluation:
        meilleur_individu = individu
        meilleure_evaluation = evaluation

print(meilleur_individu, meilleure_evaluation)
