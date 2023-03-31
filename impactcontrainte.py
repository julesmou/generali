import pandas as pd
import matplotlib.pyplot as plt

# Lire le fichier CSV
result_df = pd.read_csv('result2.csv')

# Créer une figure et des axes pour les tracés
fig, ax1 = plt.subplots()

# Tracer le gain en fonction du risque
ax1.plot(result_df['risque'], result_df['gain'], 'b-', label='Gain')
ax1.set_xlabel('Risque')
ax1.set_ylabel('Gain', color='b')
ax1.tick_params('y', colors='b')

# Créer un deuxième axe partageant le même axe x
ax2 = ax1.twinx()

# Tracer la revalorisation en fonction du risque
ax2.plot(result_df['risque'], result_df['revalorisation'], 'r-', label='Revalorisation')
ax2.set_ylabel('Revalorisation', color='r')
ax2.tick_params('y', colors='r')

# Ajouter une légende et afficher le graphique
fig.legend(loc="upper left", bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes)
plt.show()