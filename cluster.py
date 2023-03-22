# Importation des bibliothèques nécessaires
# pip install scikit-learn
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
chemin_fichier = "/Users/antbe/Downloads/ptf_mrh_2020_elast_plr.xlsx"
# Charger les données à partir du fichier Excel
data = pd.read_excel(chemin_fichier)
data['coeff_prix'] = data['coeff_prix'].multiply(5)
# Sélectionner les colonnes qui contiennent les données à clusteriser
X = data[['prime_profit', 'pcc', 'coeff_non_prix',
          'coeff_prix', 'proba_resil_0%', 'proba_resil_5%']]

# Standardiser les données pour avoir une moyenne nulle et une variance unitaire
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Supprimer les lignes contenant des valeurs manquantes
X_scaled = X_scaled[~np.isnan(X_scaled).any(axis=1)]


# Création d'un objet KMeans avec 15 clusters
kmeans = KMeans(n_clusters=15)

# Entraînement du modèle sur les données
kmeans.fit(X_scaled)

# Prédiction des clusters pour les données d'entrée
labels = kmeans.predict(X_scaled)

# Affichage des centres de cluster
print(kmeans.cluster_centers_)

# Affichage des labels de cluster prédits
print(labels)
