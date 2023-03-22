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
n = len(X_scaled[['coeff_non_prix']])

# on fait une liste de notre répartition uniforme
A = np.linspace(-0.1, 0.1, 10)
# on défini la fonction de rétention


def retention(x):
    ret = []
    m = 0
    for i in range(n):
        r = 1/(1+np.exp(X_scaled["coeff_non_prix"].iloc[i] +
               X_scaled["coeff_prix"].iloc[i]*x[i]))
        ret.append(r)  # on définit la proba de rétention pour chaque client
        m = ret[i]+m
    return ret, m/n

# on définit la fonction de gain


def gain(x):
    ret = retention(x)[0]
    res = 0
    for i in range(n):
        res = res+(X_scaled["prime_profit"].iloc[i] *
                   (1+x[i])-X_scaled['pcc'].iloc[i])*ret[i]
    return res


# on crée la liste des gains et de la rétention moyenne pour des réévaluations uniformes

Rétention = []
Gain = []
for i in range(len(A)):
    x = np.ones(n)*A[i]
    ret = retention(x)[1]
    g = gain(x)
    Rétention.append(ret)
    Gain.append(g)

#plt.plot(A, Rétention)
# plt.show()

#plt.plot(A, Gain)
# plt.show()


plt.plot(A, Rétention)
plt.xlabel('taux de réévaluation uniforme')
plt.ylabel('taux de rétention moyen')
plt.show()


plt.plot(A, Gain)
plt.xlabel('taux de réévaluation uniforme')
plt.ylabel('Gain')
plt.show()
