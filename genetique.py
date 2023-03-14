#from test import *
from data import *
import random
import numpy as np
import random as rd

args = [pcc, prime_profit, a, b]


def retention(x):
    ret = []
    m=0
    for i in range(taille_individu):
        r = 1/(1+np.exp(args[2][i]+args[3][i]*x[i]))
        ret.append(r)  # on définit la proba de rétention pour chaque client
    for i in range (taille_individu):
        m=ret[i]+m
        
    return ret,m/taille_individu
        

def fun(x, args):
    ret = retention(x)[0]
    res = 0
    for i in range(taille_individu):
        res = res+(args[1][i]*(1+x[i])-args[0][i])*ret[i]
    return res

# Fonction objectif à optimiser
def objectif(x):
    result=fun(x,args)
    return result

# Définition des paramètres de l'algorithme génétique
taille_population = 100
taille_individu = len(pcc)
nb_generations = 50
probabilite_mutation = 0.1


# Création de la population initiale
population = [] 
for i in range(taille_population):
    individu = []
    for j in range(taille_individu):
        # Génération aléatoire d'un gène
        gene=(np.random.rand(1)/5-0.1)
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
        enfant1 = parent1[:int(taille_individu/2)] + parent2[int(taille_individu/2):]
        enfant2 = parent2[:int(taille_individu/2)] + parent1[int(taille_individu/2):]
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

print(meilleur_individu,meilleure_evaluation)