# pcc=[195.1,241.9,209.5,157.9]
# prime_profit=[656.76,565.07,667.01,446.53]
# a=[-3.13,-3.65,-4.05,-3.2]
# b=[0.030,0.028,0.0077,0.011]

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# Définir le chemin du fichier CSV
chemin_fichier = "/Users/julesmouradian/Desktop/Data_Int.csv"

# Charger le fichier CSV en tant que DataFrame
DataFrame = pd.read_csv(chemin_fichier)

# Créer un dictionnaire intermédiaire à partir du DataFrame
DataDictInt = DataFrame.to_dict()
DataDict = DataDictInt["prime_profit;pcc;coeff_non_prix;coeff_prix;proba_resil_0%;proba_resil_5%"]

# transformer un ligne sur dictionnaire, qui sont des chaines de caractères, en listes


def transfo_ligne(n):
    # séparer la chaîne en une liste de nombres
    liste_nombres = DataDict[n].split(';')
    # convertir chaque élément de la liste en nombre flottant
    nombres_flottants = [float(x) for x in liste_nombres[:-2]]
    # convertir les deux derniers éléments de la liste en nombres entiers
    derniers_nombres = [float(x[:-1])/100 for x in liste_nombres[-2:]]
    DataDict[n] = nombres_flottants+derniers_nombres


# appliquer la fonction à toutes les lignes
for i in range(0, 10):
    transfo_ligne(i)

# print(DataDict)

 # 0 : prime_profit
 # 1 : pcc
 # 2 : coeff_non_prix
 # 3 : coeff_prix
 # 4 : proba_resil_0%
 # 5 : proba_resil_5%

prime_profit = []
for i in range(10):
    prime_profit.append(DataDict[i][0])

pcc = []
for i in range(10):
    pcc.append(DataDict[i][1])

coeff_non_prix = []
for i in range(10):
    coeff_non_prix.append(DataDict[i][2])

a = []
for i in range(10):
    a.append(DataDict[i][3])

b = []
for i in range(10):
    b.append(DataDict[i][4])

# proba_resil_5%=[]
# for i in range (10):
#    pcc.append(DataDict[i][5])
