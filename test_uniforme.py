from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Charger les données à partir du fichier Excel
data = pd.read_csv(
    "/Users/antbe/Downloads/ST7_projet/generali/filtre_0.01_sans_augmentation.csv")


X = data[['prime_profit', 'pcc', 'coeff_non_prix',
          'coeff_prix']]


# on récupère la longueur de la liste
taille_individu = len(X[['coeff_non_prix']])


def retention(x):
    ret = 1/(1+np.exp(X["coeff_non_prix"] + X["coeff_prix"]*x))
    somme_ret = ret.sum()
    mean_rate = somme_ret/taille_individu
    return ret, mean_rate


def fun(x):
    ret = retention(x)[0]
    marge = (X["prime_profit"] * (1+x)-X['pcc'])*ret
    res = marge.sum()
    return res


# on fait une liste de notre répartition uniforme
A = np.linspace(-0.2, 0.2, 100)
# on défini la fonction de rétention


# on crée la liste des gains et de la rétention moyenne pour des réévaluations uniformes

Rétention = []
Gain = []
for i in range(len(A)):
    x = np.ones(taille_individu)*A[i]
    ret = retention(x)[1]
    g = fun(x)
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
