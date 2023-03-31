import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# Lecture du fichier csv contenant les données
df = pd.read_csv("/Users/julesmouradian/Desktop/Centrale/Generali/ziz.csv")
Y = df[['prime_profit', 'pcc', 'coeff_non_prix', 'coeff_prix']]

# Création du nouveau dataframe
new_X = Y.apply(lambda x: pd.Series({
    'rentabilite': x['prime_profit'] / x['pcc'],
    'coeff_non_prix': x['coeff_non_prix'],
    'coeff_prix': x['coeff_prix']
}), axis=1)


# Choix du nombre de clusters à créer
n_clusters = 9

# Application de l'algorithme de clustering K-Means
kmeans = KMeans(n_clusters=n_clusters, n_init=10)
kmeans.fit(new_X)

# Affichage des résultats
new_X.loc[:, 'cluster'] = kmeans.labels_
V={}
for i in range (0,9):
    Dpp={}
    Dpcc={}
    Dpp[i]=Y[new_X['cluster']==i]['prime_profit'].sum()
    Dpcc[i]=Y[new_X['cluster']==i]['pcc'].sum()
    V[i]=[Dpp[i],Dpcc[i],kmeans.cluster_centers_[i]]


'''''''''
# Création de la figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Définissez une liste de couleurs pour chaque cluster
colors = ['r', 'g', 'b', 'y', 'c', 'm', 'orange', 'purple', 'pink']

# Tracé des points avec des couleurs de cluster
for i in range(n_clusters):
    x = new_X.loc[new_X['cluster'] == i, 'coeff_non_prix']
    y = new_X.loc[new_X['cluster'] == i, 'coeff_prix']
    z = new_X.loc[new_X['cluster'] == i, 'rentabilite']
    ax.scatter(x, y, z, c=colors[i], label=f'Cluster {i}')
'''''''''
'''''''''
# Étiquetage des axes
ax.set_xlabel('Coeff_non_prix')
ax.set_ylabel('Coeff_prix')
ax.set_zlabel('Rentabilite')

# Ajout de la légende
ax.legend()
'''''''''''
# Affichage du graphique
#plt.show()