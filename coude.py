# Importation des bibliothèques nécessaires
#pip install scikit-learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
chemin_fichier = "/Users/julesmouradian/Desktop/Centrale/donnees.generali.xltx"
# Charger les données à partir du fichier Excel
data = pd.read_excel(chemin_fichier)
data['coeff_prix'] = data['coeff_prix'].multiply(5)
from sklearn.impute import SimpleImputer
# Sélectionner les colonnes qui contiennent les données à clusteriser
X = data[[ 'prime_profit', 'pcc', 'coeff_non_prix', 'coeff_prix', 'proba_resil_0%','proba_resil_5%']]

# Standardiser les données pour avoir une moyenne nulle et une variance unitaire
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Supprimer les lignes contenant des valeurs manquantes
X_scaled = X_scaled[~np.isnan(X_scaled).any(axis=1)]
# Initialiser la somme des carrés des distances intra-cluster pour chaque nombre de clusters de 1 à 10
ssd = []
for k in range(1, 60):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_scaled)
    ssd.append(kmeans.inertia_)

# Tracer la courbe de la somme des carrés des distances intra-cluster en fonction du nombre de clusters
plt.plot(range(1, 60), ssd, 'bx-')
plt.xlabel('Nombre de clusters (k)')    
plt.ylabel('Somme des carrés des distances intra-cluster')
plt.title('Méthode du coude')
plt.show()