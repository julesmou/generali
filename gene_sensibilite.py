import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# on recrée le fichier de base à chaque fois et on fait les modifications pour avoir des données cleans

chemin_fichier = "/Users/antbe/Downloads/ptf_mrh_2020_elast_plr.xlsx"

# Lecture du fichier Excel contenant les données
df = pd.read_excel(chemin_fichier)

# Sélection des colonnes à utiliser pour le clustering
Y = df[['prime_profit', 'pcc', 'coeff_non_prix', 'coeff_prix']]

# Standardiser les données pour avoir une moyenne nulle et une variance unitaire
newY = pd.DataFrame(Y)

# Supprimer les lignes contenant des valeurs manquantes
newY = newY.dropna()

# Multiplier la colonne 'coeff_prix' par 5
newY['coeff_prix'] = newY['coeff_prix']*50

# Supprimer les lignes où 'coeff_prix' est négatif
newY = newY.loc[newY['coeff_prix'] >= 0]

# on ajoute 10% du coeff non prix > translation
n = len(newY[['coeff_non_prix']])
mean_a = newY[['coeff_non_prix']].sum()/n
newY[['coeff_non_prix']] = newY[['coeff_non_prix']]-0.1*mean_a


# on ajoute 10% du coeff prix > translation
#mean_b = newY[['coeff_prix']].sum()/n
#newY[['coeff_prix']] = newY[['coeff_prix']]+mean_b*0.1


# Définir les quantiles d'ordre α et 1 - α
alpha = 0.01

# Calculer les quantiles d'ordre α et 1 - α pour chaque colonne
lower_quantiles = newY.quantile(alpha)
upper_quantiles = newY.quantile(1 - alpha)

# Supprimer les données inférieures au quantile d'ordre α et supérieures au quantile d'ordre 1 - α
filtered_newY = newY.copy()
for col in newY.columns:
    filtered_newY = filtered_newY.loc[(filtered_newY[col] >= lower_quantiles[col]) & (
        filtered_newY[col] <= upper_quantiles[col])]

filtered_newY.to_csv('filtre_0.01_avec_augmentation_50.csv', index=False)
