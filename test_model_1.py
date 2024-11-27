import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Simuler des données historiques de temps d'attente
def generate_historical_data(days=30):
    times = pd.date_range("11:30", "13:30", freq="10min")
    data = []
    for day in range(days):
        base_wait = np.random.randint(5, 15)  # Temps moyen de base pour la journée
        wait_times = [
            max(0, base_wait + np.random.randint(-5, 5) + (5 if 12 <= t.hour < 13 else 0))
            for t in times
        ]
        for t, w in zip(times, wait_times):
            data.append([day, t.time(), w])
    return pd.DataFrame(data, columns=["day", "time", "wait_time"])

# Génération des données historiques
data = generate_historical_data()

# Transformation des données pour la prédiction
# Transformation des données pour la prédiction
def prepare_data(data):
    # Pivot pour aligner les temps par colonne (un jour = une ligne)
    pivot = data.pivot(index="day", columns="time", values="wait_time")
    
    # Renommer les colonnes en chaînes pour éviter les problèmes de type
    pivot.columns = pivot.columns.astype(str)
    
    # Ajouter des décalages pour les prévisions
    for col in pivot.columns:
        pivot[f"{col}_lag1"] = pivot[col].shift(1)  # Décalage de 1 jour
    
    # Supprimer les lignes avec NaN dues au décalage
    pivot = pivot.dropna()
    
    return pivot

# Préparer les données
processed_data = prepare_data(data)

# Séparer les caractéristiques et la variable cible
X = processed_data[[col for col in processed_data.columns if "lag1" in col]]
y = processed_data[[col for col in processed_data.columns if "lag1" not in col]]


# Préparer les données
processed_data = prepare_data(data)

# Séparation des caractéristiques et de la variable cible
X = processed_data[[col for col in processed_data.columns if "lag1" in col]]
y = processed_data[[col for col in processed_data.columns if "lag1" not in col]]

# Diviser en ensembles d'entraînement/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un modèle de régression
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédire sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer le modèle
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

# Faire des prévisions pour un nouveau jour
def predict_for_today(model, recent_wait_times):
    recent_wait_times = recent_wait_times.reshape(1, -1)  # Redimensionner pour l'entrée
    return model.predict(recent_wait_times)

# Exemple de prédiction pour un nouveau jour
new_day_input = X.iloc[-1].values  # Les derniers temps d'attente connus (jour précédent)
predictions = predict_for_today(model, new_day_input)
print("Prévisions des temps d'attente :", predictions)
