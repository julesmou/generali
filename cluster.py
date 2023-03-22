import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
chemin_fichier = "/Users/julesmouradian/Desktop/Centrale/donnees.generali.xltx"
# Lecture du fichier Excel contenant les données
df = pd.read_excel(chemin_fichier)

# Sélection des colonnes à utiliser pour le clustering
X = df[[ 'coeff_non_prix', 'coeff_prix']]
Y= df[['prime_profit','pcc','coeff_non_prix', 'coeff_prix']]
# Standardiser les données pour avoir une moyenne nulle et une variance unitaire
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)
newX_scaled = pd.DataFrame(X)
newY= pd.DataFrame(Y)
# Supprimer les lignes contenant des valeurs manquantes
newX_scaled = newX_scaled.dropna()
newY=newY.dropna()
# Choix du nombre de clusters à créer
n_clusters = 9

# Application de l'algorithme de clustering K-Means
kmeans = KMeans(n_clusters=n_clusters, n_init=10)
kmeans.fit(newX_scaled)

# Affichage des résultats
#print(kmeans.labels_)
newX_scaled['cluster']=kmeans.labels_
#print(newX_scaled)
V={}
for i in range (0,9):
    Dpp={}
    Dpcc={}
    Dpp[i]=newY[newX_scaled['cluster']==i]['prime_profit'].sum()
    Dpcc[i]=newY[newX_scaled['cluster']==i]['pcc'].sum()
    V[i]=[Dpp[i],Dpcc[i],kmeans.cluster_centers_[i]]
print(V)
# Plot the data colored by cluster label
#plt.scatter(newX_scaled.iloc[:,0], newX_scaled.iloc[:,1], c=kmeans.labels_)
#plt.xlabel('Première variable')
#plt.ylabel('Deuxième variable')
#plt.title('Résultats du clustering avec K-Means')
#plt.show()





