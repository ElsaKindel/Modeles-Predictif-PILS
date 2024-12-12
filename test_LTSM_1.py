import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def generate_wait_time(mu = 40, show_gaussian = False):
    """
    mu : (int, default 40) Heure moyenne du pic (40e minute, soit 12h10)
    show_gaussian : (boolean, default False) Permet de visualiser la répartition du temps d'attente

    return : wait_times (array)
    """
    # Paramètres de la simulation
    time_frame = 120 # en minutes, durée du service 
    min_per_sample = 5 #fréquence des échantillons en minute (1 échantillon = 5 minutes)
    n_intervals = int(time_frame/min_per_sample)  # 12 points pour une plage de 2 heures (chaque point = 10 minutes)
    ech_mu = mu/min_per_sample #pic rammené au nombre d'échantillons
    sigma = 10  # Écart-type pour modéliser la montée et la descente progressive
    base_attente = 1  # Attente minimale (en minutes)
    fluctuation_scale = 3  # Amplitude des fluctuations (bruit rouge)
    amplitude_attente = 20 # Durée maximale d'attente

    # Étape 1 : Générer une tendance globale avec une gaussienne
    x = np.arange(0, time_frame-1, min_per_sample)  # Minutes de la plage horaire (start, stop, step)
    x_hours = ["11:30", "11:40", "11:50", "12", "12:10","12:20","12:30", "12:40", "12:50",  "13",  "13:10",  "13:20"]
    trend = base_attente + amplitude_attente * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    # Étape 2 : Générer des fluctuations cohérentes avec un bruit rouge
    noise = [0]  # Initialiser avec une valeur de départ
    for _ in range(1, n_intervals):
        next_noise = noise[-1] + np.random.normal(0, fluctuation_scale)
        noise.append(next_noise)

    # Normaliser le bruit pour qu'il oscille autour de 0 avec une amplitude contrôlée
    noise = np.array(noise)
    noise = (noise - noise.mean()) / noise.std() * fluctuation_scale

    # Étape 3 : Ajouter le bruit rouge à la tendance globale
    wait_times = trend + noise

    # Étape 4 : S'assurer que les temps d'attente restent positifs
    wait_times = np.clip(wait_times, 0, None)

    # Visualiser les données générées
    if show_gaussian :
        plt.figure(figsize=(10, 6))
        plt.plot(x, trend, label="Tendance globale (gaussienne)", linestyle="--")
        plt.plot(x, noise, label="Bruit rouge (fluctuations)", linestyle=":")
        plt.plot(x, wait_times, label="Temps d'attente générés (hybride)", color="blue")
        plt.xlabel("Temps (minutes après 11h30)")
        plt.xticks(np.arange(0, time_frame-1, 10), x_hours)
        plt.ylabel("Temps d'attente (minutes)")
        plt.title("Temps d'attente générés : Tendance + Bruit Rouge")
        plt.legend()
        plt.show()

    return wait_times

def generate_historical_data(num_days=30):
    """
    Fonction pour générer des données historiques simulées pour les temps d'attente.
    
    :param num_days: Le nombre de jours pour lesquels générer les données.
    :return: Un DataFrame contenant les données simulées.
    """

    # Liste des heures de la journée entre 11h30 et 13h20 (par exemple, tous les 10 minutes)
    time_intervals = pd.date_range("11:30", "13:25", freq="5min").strftime('%H:%M').tolist()
    
    # Initialisation de la liste pour stocker les données
    data = pd.DataFrame()
    
    # Générer des données pour chaque jour
    for day_offset in range(num_days):
        # Calculer la date du jour actuel
        current_day = (datetime.now() - timedelta(days=day_offset)).strftime('%Y-%m-%d')
        
        # Générer des temps d'attente aléatoires (par exemple entre 0 et 20 minutes)
        # Vous pouvez modifier cette partie pour ajouter un comportement plus réaliste
        #for time in time_intervals:
            # Simuler un temps d'attente aléatoire (par exemple, entre 0 et 20 minutes)
        wait_time = generate_wait_time(show_gaussian=False)#np.random.uniform(0, 20) #generate_wait_time()
        # Ajouter la ligne de données au tableau
        current_data = {"day" : np.full(len(time_intervals), current_day), "time" : time_intervals, "wait_time" : wait_time}
        current_data_df = pd.DataFrame(current_data, columns=["day", "time","wait_time"])
        #data.append(current_data)#data.append([current_day, time, wait_time])
        data = pd.concat([data, current_data_df], ignore_index=True)

    # Créer un DataFrame à partir des données
    #df = pd.DataFrame.from_dict(data)
    
    return data


# Simuler des données historiques
data = generate_historical_data(num_days=30)
print(data)

# Préparer les données pour un modèle LSTM
def prepare_lstm_data(data, lookback=7):
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
lookback = 7  # Utiliser les données des 7 derniers jours
X, y, scaler = prepare_lstm_data(data, lookback=lookback)

# Diviser en ensemble d'entraînement et de test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Construire le modèle LSTM
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(y_train.shape[1])
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Entraîner le modèle
history = model.fit(X_train, y_train, epochs=60, batch_size=16, validation_data=(X_test, y_test))
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

# Évaluer les performances
mae = np.mean(np.abs(scaler.inverse_transform(y_test) - y_pred_rescaled))
print(f"Mean Absolute Error: {mae:.2f}")


