import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

chemin_fichier = "/Users/julesmouradian/Desktop/Centrale/donnees.generali.xltx"

# Lecture du fichier Excel contenant les données
df = pd.read_excel(chemin_fichier)

# Sélection des colonnes à utiliser pour le clustering
Y = df[['prime_profit', 'pcc', 'coeff_non_prix', 'coeff_prix']]

# Standardiser les données pour avoir une moyenne nulle et une variance unitaire
newY = pd.DataFrame(Y)

# Supprimer les lignes contenant des valeurs manquantes
newY = newY.dropna()

# Multiplier la colonne 'coeff_prix' par 5
newY['coeff_prix'] = newY['coeff_prix'].multiply(5)

# Supprimer les lignes où 'coeff_prix' est négatif
newY = newY.loc[newY['coeff_prix'] >= 0]
# Définir les quantiles d'ordre α et 1 - α
alpha = 0.05

# Calculer les quantiles d'ordre α et 1 - α pour chaque colonne
lower_quantiles = newY.quantile(alpha)
upper_quantiles = newY.quantile(1 - alpha)

# Supprimer les données inférieures au quantile d'ordre α et supérieures au quantile d'ordre 1 - α
filtered_newY = newY.copy()
for col in newY.columns:
    filtered_newY = filtered_newY.loc[(filtered_newY[col] >= lower_quantiles[col]) & (filtered_newY[col] <= upper_quantiles[col])]

filtered_newY.to_csv('ziz.csv',index=False)
