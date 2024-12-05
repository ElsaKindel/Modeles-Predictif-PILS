import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def generate_wait_time(mu = 40, show_gaussian = False):
    """
    mu : (int, default 40) Heure moyenne du pic (40e minute, soit 12h10)
    show_gaussian : (boolean, default False) Permet de visualiser la répartition du temps d'attente

    return : wait_times (array)
    """
    # Paramètres de la simulation
    time_frame = 120 # en minutes, durée du service 
    min_per_sample = 5 #fréquence des échantillons en minute (1 échantillon = 10 minutes)
    n_intervals = int(time_frame/min_per_sample)  # 12 points pour une plage de 2 heures (chaque point = 10 minutes)
    ech_mu = mu/min_per_sample #pic rammené au nombr d'échantillons
    sigma = 20  # Écart-type pour modéliser la montée et la descente progressive
    base_attente = 1  # Attente minimale (en minutes)
    fluctuation_scale = 3  # Amplitude des fluctuations (bruit rouge)
    amplitude_attente = 20 # Durée maximale d'attente

    # Étape 1 : Générer une tendance globale avec une gaussienne
    x = np.arange(0, time_frame-1, min_per_sample)  # Minutes de la plage horaire (start, stop, step)
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
        plt.ylabel("Temps d'attente (minutes)")
        plt.title("Temps d'attente générés : Tendance + Bruit Rouge")
        plt.legend()
        plt.show()

    return wait_times

def generate_historical_data(num_days=30):
    """
    Génère des données historiques de temps d'attente en utilisant une gaussienne 
    pour simuler les variations journalières et exclut les week-ends.
    
    :param num_days: Nombre de jours ouvrés pour lesquels générer les données.
    :param show_gaussian: Permet de visualiser les courbes générées pour chaque jour.
    :return: Un DataFrame contenant les données simulées.
    """
    # Liste des heures de la journée (toutes les 10 minutes entre 11h30 et 13h30)
    time_intervals = pd.date_range("11:30", "13:30", freq="5min").strftime('%H:%M').tolist()
    
    # Initialisation des données
    data = []
    days_generated = 0
    current_date = datetime.now()

    while days_generated < num_days:
        # Exclure les week-ends
        if current_date.weekday() >= 5:  # 5 = Samedi, 6 = Dimanche
            current_date -= timedelta(days=1)
            continue
        
        # Simuler les temps d'attente pour ce jour
        wait_times = generate_wait_time(show_gaussian=False)
        
        # Ajouter les données pour chaque tranche horaire
        for time, wait_time in zip(time_intervals, wait_times):
            data.append([current_date.strftime('%Y-%m-%d'), time, wait_time])
        
        # Passer au jour précédent
        current_date -= timedelta(days=1)
        days_generated += 1

    # Créer un DataFrame
    df = pd.DataFrame(data, columns=["day", "time", "wait_time"])
    
    return df

# Simuler des données historiques
data = generate_historical_data(num_days=30)
print(data)

def prepare_lstm_data_weekly_pattern(data, lookback_weeks=4):
    pivot = data.pivot(index="day", columns="time", values="wait_time")
    values = pivot.values
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)
    
    X, y = [], []
    for i in range(lookback_weeks, len(scaled_values)):
        X.append(scaled_values[i - lookback_weeks:i, :])
        y.append(scaled_values[i, :])
    
    # Vérification des formes avant de retourner
    print(f"Longueur de X : {len(X)}")
    print(f"Longueur de y : {len(y)}")
    if len(X) == 0 or len(y) == 0:
        raise ValueError("X ou y est vide. Vérifiez les paramètres de lookback_weeks ou les données d'entrée.")

    return np.array(X), np.array(y), scaler

# Préparer les données
X, y, scaler = prepare_lstm_data_weekly_pattern(data, lookback_weeks=4)

# Vérifier les dimensions après conversion
print(f"Forme finale de X : {X.shape}")  # Doit être (num_samples, lookback_weeks, num_features)
print(f"Forme finale de y : {y.shape}")  # Doit être (num_samples, num_features)


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