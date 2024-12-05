import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def generate_historical_data(num_days=30):
    """
    Génère des données historiques réalistes pour les temps d'attente,
    en excluant les week-ends (samedi et dimanche).
    
    :param num_days: Nombre de jours ouvrés pour lesquels générer les données.
    :return: Un DataFrame contenant les données simulées.
    """
    # Liste des heures de la journée (toutes les 10 minutes entre 11h30 et 13h30)
    time_intervals = pd.date_range("11:30", "13:30", freq="10min").strftime('%H:%M').tolist()
    
    # Moyennes et écarts-types ajustés aux heures (comportement horaire typique)
    hourly_means = {
        "11:30": 1, "11:40": 3, "11:50": 4,
        "12:00": 10, "12:10": 15, "12:20": 20,
        "12:30": 20, "12:40": 16, "12:50": 14,
        "13:00": 10, "13:10": 5, "13:20": 3, "13:30": 1
    }
    hourly_std_devs = {
        "11:30": 1, "11:40": 2, "11:50": 5,
        "12:00": 10, "12:10": 10, "12:20": 5,
        "12:30": 4, "12:40": 4, "12:50": 3,
        "13:00": 3, "13:10": 2, "13:20": 2, "13:30": 1
    }
    
    # Générer les données pour chaque jour
    data = []
    days_generated = 0  # Compte les jours ouvrés générés
    current_date = datetime.now()
    
    while days_generated < num_days:
        # Vérifier si le jour actuel est un week-end
        if current_date.weekday() >= 5:  # 5 = Samedi, 6 = Dimanche
            current_date -= timedelta(days=1)
            continue
        
        # Ajouter les données horaires pour ce jour
        for time in time_intervals:
            mean = hourly_means[time]
            std_dev = hourly_std_devs[time]
            
            # Générer un temps d'attente basé sur une distribution normale
            wait_time = np.random.normal(loc=mean, scale=std_dev)
            wait_time = max(0, wait_time)  # Temps d'attente non négatif
            
            # Ajouter la ligne de données
            data.append([current_date.strftime('%Y-%m-%d'), time, wait_time])
        
        # Passer au jour précédent
        current_date -= timedelta(days=1)
        days_generated += 1
    
    # Créer un DataFrame à partir des données
    df = pd.DataFrame(data, columns=["day", "time", "wait_time"])
    
    return df


# Simuler des données historiques
data = generate_historical_data(num_days=30)
print(data)

# Préparer les données pour un modèle LSTM
def prepare_lstm_data(data, lookback= 17):
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
lookback = 17  # Utiliser les données des X derniers jours
X, y, scaler = prepare_lstm_data(data, lookback=lookback)

# Diviser en ensemble d'entraînement et de test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Construire le modèle LSTM
model = Sequential([
    #Bidirectional(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))),
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(y_train.shape[1])
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Entraîner le modèle
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

#Plot graphe loss en fonction des epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Courbes de Perte (Loss) pendant l'Entraînement")
plt.xlabel("Épochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.show()

# Prédire pour un nouveau jour
y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)  # Revenir aux valeurs originales
print("Prédiction du jour", y_pred_rescaled)

# Évaluer les performances
mae = np.mean(np.abs(scaler.inverse_transform(y_test) - y_pred_rescaled))
print(f"Mean Absolute Error: {mae:.2f}")