import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_historical_data(num_days=30):
    """
    Fonction pour générer des données historiques simulées pour les temps d'attente.
    
    :param num_days: Le nombre de jours pour lesquels générer les données.
    :return: Un DataFrame contenant les données simulées.
    """
    # Liste des heures de la journée entre 11h30 et 13h30 (par exemple, tous les 10 minutes)
    time_intervals = pd.date_range("11:30", "13:30", freq="10min").strftime('%H:%M').tolist()
    
    # Initialisation de la liste pour stocker les données
    data = []
    
    # Générer des données pour chaque jour
    for day_offset in range(num_days):
        # Calculer la date du jour actuel
        current_day = (datetime.now() - timedelta(days=day_offset)).strftime('%Y-%m-%d')
        
        # Générer des temps d'attente aléatoires (par exemple entre 0 et 20 minutes)
        # Vous pouvez modifier cette partie pour ajouter un comportement plus réaliste
        for time in time_intervals:
            # Simuler un temps d'attente aléatoire (par exemple, entre 0 et 20 minutes)
            wait_time = np.random.uniform(0, 20)
            
            # Ajouter la ligne de données au tableau
            data.append([current_day, time, wait_time])
    
    # Créer un DataFrame à partir des données
    df = pd.DataFrame(data, columns=["day", "time", "wait_time"])
    
    return df


# Simuler des données historiques
data = generate_historical_data(num_days=30)
print(data.head())

# Préparer les données pour un modèle LSTM
def prepare_lstm_data(data, lookback=3):
    pivot = data.pivot(index="day", columns="time", values="wait_time")
    values = pivot.values
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)
    
    X, y = [], []
    for i in range(lookback, len(scaled_values)):
        X.append(scaled_values[i - lookback:i, :])
        y.append(scaled_values[i, :])
    return np.array(X), np.array(y), scaler

# Préparer les données pour LSTM
lookback = 3  # Utiliser les données des 3 derniers jours
X, y, scaler = prepare_lstm_data(data, lookback=lookback)

# Diviser en ensemble d'entraînement et de test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Construire le modèle LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(y_train.shape[1])
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Entraîner le modèle
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Prédire pour un nouveau jour
y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)  # Revenir aux valeurs originales

# Évaluer les performances
mae = np.mean(np.abs(scaler.inverse_transform(y_test) - y_pred_rescaled))
print(f"Mean Absolute Error: {mae:.2f}")


