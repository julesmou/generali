import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# Lecture du fichier csv contenant les données
df = pd.read_csv("/Users/julesmouradian/Desktop/Centrale/Generali/ziz.csv")

# Sélection des colonnes à utiliser pour le clustering
X = df[[ 'coeff_non_prix', 'coeff_prix']]
Y= df[['prime_profit','pcc','coeff_non_prix', 'coeff_prix']]
# Standardiser les données pour avoir une moyenne nulle et une variance unitaire


# Choix du nombre de clusters à créer
n_clusters = 9

# Application de l'algorithme de clustering K-Means
kmeans = KMeans(n_clusters=n_clusters, n_init=10)
kmeans.fit(X)

# Affichage des résultats
#print(kmeans.labels_)
X['cluster']=kmeans.labels_
V={}
for i in range (0,9):
    Dpp={}
    Dpcc={}
    Dpp[i]=Y[X['cluster']==i]['prime_profit'].sum()
    Dpcc[i]=Y[X['cluster']==i]['pcc'].sum()
    V[i]=[Dpp[i],Dpcc[i],kmeans.cluster_centers_[i]]
print(V)






