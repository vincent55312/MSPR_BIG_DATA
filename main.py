import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Charger le fichier Excel
fichier_excel = './data/c.xlsx'
df = pd.read_excel(fichier_excel)

# Convertir les chaînes avec des virgules en flottants pour toutes les colonnes concernées
for parti in ['Voix % D', 'Voix % G', 'Voix % C', 'Voix % ED', '% Chomage']:
    df[parti] = df[parti].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)

# Convertir les années en format numpy pour l'entrainement
X = df['Année'].values.reshape(-1, 1)

# Étendre la plage des années jusqu'en 2027 pour la prédiction
X_future = np.arange(X.min(), 2027 + 1).reshape(-1, 1)

# Créer et entraîner un modèle de régression linéaire pour chaque parti et pour le chômage
partis = ['Voix % D', 'Voix % G', 'Voix % C', 'Voix % ED']  # Retirer % Chomage de la liste pour le traiter séparément

plt.figure(figsize=(10, 6))  # Définir la taille de la figure

ax1 = plt.gca()  # Obtenir l'axe actuel
ax2 = ax1.twinx()  # Créer un deuxième axe des y qui partage le même axe des x

for parti in partis:
    y = df[parti].values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    y_pred_future = model.predict(X_future)
    
    ax1.scatter(X, y, label=f'Données réelles {parti}')
    ax1.plot(X_future, y_pred_future, label=f'Prédiction jusqu\'en 2027 {parti}')

# Traiter les données du chômage séparément
y_chomage = df['% Chomage'].values.reshape(-1, 1)
model_chomage = LinearRegression()
model_chomage.fit(X, y_chomage)
y_pred_chomage = model_chomage.predict(X_future)

ax2.scatter(X, y_chomage, color='purple', label='Données réelles % Chomage')
ax2.plot(X_future, y_pred_chomage, color='purple', label='Prédiction jusqu\'en 2027 % Chomage')

ax1.set_xlabel('Année')
ax1.set_ylabel('Pourcentage de voix')
ax2.set_ylabel('% Chomage', color='purple')  # Définir la couleur de l'axe des y pour le chômage

plt.title('Régression linéaire et prédiction jusqu\'en 2027 pour chaque parti et % Chomage')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()