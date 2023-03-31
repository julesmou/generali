import matplotlib.pyplot as plt
from cluster import *
import numpy as np
from scipy.optimize import minimize
from random import *
import matplotlib.pyplot as plt
from slsqp import *
n = 9
def sensitivity_analysis(choc_prime, choc_elasticity, pcc, prime_profit, a, b):
    pcc_shock = [prime * (1 + choc_prime) for prime in pcc]
    b_shock = [coef * (1 + choc_elasticity) for coef in b]
    result_shock = minimize(fun, x0, args=(pcc_shock, prime_profit, a, b_shock), method='SLSQP', bounds=bounds, constraints=constraints)
    # Calculate the average retention
    avg_retention = retention(result_shock.x)[1]
    
    while avg_retention < 0.93:
        pcc_shock = [prime * (1 + choc_prime) for prime in pcc]
        result_shock = minimize(fun, x0, args=(pcc_shock, prime_profit, a, b_shock), method='SLSQP', bounds=bounds, constraints=constraints)
        
        # Calculate the average retention
        avg_retention = retention(result_shock.x)[1]
    return(result_shock)

# Choc aléatoire de +/- 5% sur la prime technique
choc_prime = uniform(-0.05, 0.05)

# Choc aléatoire sur le coefficient d'élasticité au prix
choc_elasticity_values = [uniform(-0.1, 0.1) for _ in range(5)]  # Vous pouvez changer le nombre de valeurs testées en modifiant la valeur 5

# Appliquer les chocs et analyser l'impact sur le portefeuille optimal
result_with_shocks = []
for choc_elasticity in choc_elasticity_values:
    result_shock = sensitivity_analysis(choc_prime, choc_elasticity, pcc, prime_profit, a, b)
    result_with_shocks.append(result_shock)

# Appliquer les chocs séparément et analyser l'impact sur le portefeuille optimal
result_with_prime_shock = sensitivity_analysis(choc_prime, 0, pcc, prime_profit, a, b)
results_with_elasticity_shocks = [sensitivity_analysis(0, choc_elasticity, pcc, prime_profit, a, b) for choc_elasticity in choc_elasticity_values]

# Afficher les résultats des tests de sensibilité
print("Résultats des tests de sensibilité :")

print(f"Choc sur la prime technique: {choc_prime:.4f}")
print(f"Optimisation après choc: {-result_with_prime_shock.fun:.4f}")
print("Solution :", result_with_prime_shock.x)
print("")

for i, res in enumerate(results_with_elasticity_shocks):
    print(f"Choc élasticité {i + 1}: {choc_elasticity_values[i]:.4f}")
    print(f"Optimisation après choc: {-res.fun:.4f}")
    print("Solution :", res.x)
    print("")

# Comparer les résultats avec le portefeuille optimal initial
result_initial = minimize(fun, x0, args=(pcc, prime_profit, a, b), method='SLSQP', bounds=bounds, constraints=constraints)
print("Résultat initial (sans chocs) :")
print(f"Optimisation : {-result_initial.fun:.4f}")
print("Solution :", result_initial.x)

# Créer une liste contenant les gains pour chaque scénario
gains = [-result_initial.fun] + [-result_with_prime_shock.fun] + [-res.fun for res in results_with_elasticity_shocks]

# Créer une liste contenant les étiquettes pour chaque scénario
labels = ['Initial'] + [f'Choc prime {choc_prime:.4f}'] + [f'Choc élasticité {i + 1}: {choc_elasticity_values[i]:.4f}' for i in range(len(choc_elasticity_values))]

# Créer un histogramme avec les gains et les étiquettes
plt.bar(labels, gains)
plt.ylabel('Gains')
plt.xticks(rotation=45, ha='right')
plt.title("Comparaison des gains pour différents scénarios d'analyse de sensibilité")
plt.show()