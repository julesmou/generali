# Importation des bibliothèques nécessaires
#pip install scikit-learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

chemin_fichier = "/Users/julesmouradian/Desktop/Centrale/donnees.generali.xltx"
# Charger les données à partir du fichier Excel
data = pd.read_excel(chemin_fichier)
data['coeff_prix'] = data['coeff_prix'].multiply(5)
# Sélectionner les colonnes qui contiennent les données à clusteriser
X = data[[ 'coeff_non_prix', 'coeff_prix']]

# Standardiser les données pour avoir une moyenne nulle et une variance unitaire
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)
newX = pd.DataFrame(X)
# Supprimer les lignes contenant des valeurs manquantes
newX = newX.dropna()
# Initialiser la somme des carrés des distances intra-cluster pour chaque nombre de clusters de 1 à 10
ssd = []
for k in range(1, 60):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(newX)
    ssd.append(kmeans.inertia_)

# Tracer la courbe de la somme des carrés des distances intra-cluster en fonction du nombre de clusters
plt.plot(range(1, 60), ssd, 'bx-')
plt.xlabel('Nombre de clusters (k)')    
plt.ylabel('Somme des carrés des distances intra-cluster')
plt.title('Méthode du coude')
plt.show()