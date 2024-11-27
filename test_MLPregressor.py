import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

# Génération de données fictives (syntétiques) pour l'exemple
# Supposons que chaque entrée a 10 caractéristiques d'entrée et qu'on prédit 12 temps d'attente
np.random.seed(42)  # Fixe la graine pour la reproductibilité
X = np.random.rand(1000, 10)  # 1000 échantillons, 10 features (ex : heure, jour, etc.)
y = np.random.rand(1000, 12)  # 1000 échantillons, 12 valeurs cibles (temps d'attente pour 12 intervalles)

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle de réseau de neurones fully connected
model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),  # 3 couches cachées avec 128, 64, et 32 neurones
    activation='relu',                # Fonction d'activation : ReLU
    solver='adam',                    # Optimiseur Adam
    max_iter=500,                     # Nombre d'itérations maximum
    random_state=42                   # Pour reproductibilité
)

# Entraînement du modèle
model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluation du modèle
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE) : {mae:.2f}")

# Exemple de prédiction pour un nouvel échantillon
nouvel_echantillon = np.random.rand(1, 10)  # Génère un échantillon aléatoire avec 10 features
predictions = model.predict(nouvel_echantillon)
print(f"Prédictions pour les 12 intervalles futurs : {predictions[0]}")
